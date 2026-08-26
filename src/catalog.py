from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio

import config


@dataclass
class SceneEntry:
    scene_id: str
    path: str
    kind: str
    aoi: str
    slot: str
    bands: list[str]
    shape: tuple
    crs: str
    bounds: tuple
    source: str


def _classify(stem: str) -> tuple[str, str, str, str] | None:
    if stem.endswith("_optical_t1"):
        return stem.split("_")[0], "optical", "t1", "synthetic"
    if stem.endswith("_optical_t2"):
        return stem.split("_")[0], "optical", "t2", "synthetic"
    if stem.endswith("_optical_stack"):
        return stem.split("_")[0], "optical", "t1", "planetary-computer"
    if stem.endswith("_sar_stack"):
        return stem.split("_")[0], "sar", "sar", "planetary-computer"
    if stem.endswith("_sar"):
        return stem.split("_")[0], "sar", "sar", "synthetic"
    return None


def _bounds(transform, width: int, height: int) -> tuple[float, float, float, float]:
    xs = [transform.c, transform.c + width * transform.a]
    ys = [transform.f, transform.f + height * transform.e]
    west, east = min(xs), max(xs)
    south, north = min(ys), max(ys)
    return (round(west, 5), round(south, 5), round(east, 5), round(north, 5))


def scan(force: bool = False) -> list[SceneEntry]:
    entries: list[SceneEntry] = []
    for p in sorted(config.RAW_DIR.glob("*.tif")):
        parsed = _classify(p.stem)
        if parsed is None:
            continue
        aoi, kind, slot, source = parsed
        try:
            with rasterio.open(p) as src:
                bands = [b.lower() for b in (src.descriptions or [])]
                shape = (src.count, src.height, src.width)
                crs = str(src.crs)
                bounds = _bounds(src.transform, src.width, src.height)
        except Exception:
            continue
        entries.append(SceneEntry(
            scene_id=p.stem, path=str(p), kind=kind, aoi=aoi, slot=slot,
            bands=bands, shape=shape, crs=crs, bounds=bounds, source=source,
        ))
    return entries


def group_by_aoi(entries: list[SceneEntry]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for e in entries:
        g = grouped.setdefault(e.aoi, {})
        g.setdefault(e.slot, e)
        g.setdefault("_source", e.source)
    return dict(sorted(grouped.items()))


def _thumb_path(entry: SceneEntry) -> Path:
    mtime = int(Path(entry.path).stat().st_mtime)
    token = hashlib.md5(f"{entry.path}_{mtime}".encode()).hexdigest()[:12]
    return config.THUMB_DIR / f"{entry.scene_id}_{token}.png"


def thumbnail_bytes(entry: SceneEntry, width: int = 360) -> bytes:
    cache = _thumb_path(entry)
    if cache.exists():
        return cache.read_bytes()
    h_target = max(96, int(width * 0.75))
    with rasterio.open(entry.path) as src:
        scale = min(1.0, width / src.width)
        out_h, out_w = max(64, int(src.height * scale)), width
        data = src.read(out_shape=(src.count, out_h, out_w)).astype(np.float32)
        names = [b.lower() for b in (src.descriptions or [])]
    if "red" in names and "green" in names and "blue" in names:
        idx = [names.index("blue"), names.index("green"), names.index("red")]
    elif data.shape[0] >= 3:
        idx = [0, 1, 2]
    else:
        idx = [0]
    chans = []
    for i in idx:
        b = data[i]
        lo, hi = np.percentile(b[np.isfinite(b)], [2, 98]) if np.isfinite(b).any() else (0, 1)
        chans.append(np.clip((b - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8))
    rgb = np.dstack(chans) if len(chans) == 3 else np.dstack([chans[0]] * 3)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if ok:
        config.THUMB_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(buf.tobytes())
        return buf.tobytes()
    raise RuntimeError("thumbnail encoding failed")
