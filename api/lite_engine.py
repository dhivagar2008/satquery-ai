from __future__ import annotations

import base64
import io
import math
import re
from typing import Any

import numpy as np
import tifffile
from PIL import Image


def _percentile_norm(band: np.ndarray, lo_p: float = 2.0, hi_p: float = 98.0) -> np.ndarray:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [lo_p, hi_p])
    if hi <= lo + 1e-9:
        return np.zeros(band.shape, dtype=np.uint8)
    return np.clip((band - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def read_tiff(data: bytes) -> dict[str, Any]:
    bio = io.BytesIO(data)
    with tifffile.TiffFile(bio) as tif:
        page = tif.pages[0]
        arr = page.asarray().astype(np.float32)
        spp = int(page.samplesperpixel or 1)

        parsed: dict[int, str] = {}
        meta_tag = page.tags.get("GDAL_METADATA")
        if meta_tag is not None and meta_tag.value:
            for sample_idx, name in re.findall(
                r'<Item name="DESCRIPTION" sample="(\d+)"[^>]*>([^<]+)</Item>',
                str(meta_tag.value),
            ):
                parsed[int(sample_idx)] = name.strip().lower()

        tags: dict[str, Any] = {}
        for tag_name in ("ModelPixelScaleTag", "ModelTiepointTag"):
            t = page.tags.get(tag_name)
            if t is not None and t.value is not None:
                tags[tag_name] = list(t.value)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3 and arr.shape[-1] == spp and arr.shape[0] != spp:
        arr = np.moveaxis(arr, -1, 0)

    n_bands = arr.shape[0]
    known = {"blue", "green", "red", "nir", "vv", "vh", "swir"}
    band_names = []
    for i in range(n_bands):
        nm = parsed.get(i, "")
        band_names.append(nm if nm in known else f"b{i}")

    if not any(b in known for b in band_names):
        band_names = (
            ["blue", "green", "red", "nir"][:n_bands] if n_bands == 4
            else ["vv", "vh"][:n_bands] if n_bands == 2
            else [f"b{i}" for i in range(n_bands)]
        )

    h, w = arr.shape[1], arr.shape[2]

    scale_deg = None
    lat_center = None
    if "ModelPixelScaleTag" in tags and len(tags["ModelPixelScaleTag"]) >= 2:
        sx, sy = float(tags["ModelPixelScaleTag"][0]), float(tags["ModelPixelScaleTag"][1])
        scale_deg = (sx, sy)
        if "ModelTiepointTag" in tags and len(tags["ModelTiepointTag"]) >= 6:
            lat_center = float(tags["ModelTiepointTag"][4])

    return {
        "bands": {name: arr[i] for i, name in enumerate(band_names)},
        "names": band_names,
        "shape": (int(h), int(w)),
        "scale_deg": scale_deg,
        "lat": lat_center,
    }


def pixel_km2(scene: dict) -> float | None:
    if scene.get("scale_deg") is None:
        return None
    sx, sy = scene["scale_deg"]
    lat = abs(scene.get("lat") or 13.0)
    m_per_deg_lat = 110540.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
    return (sx * m_per_deg_lon) * (sy * m_per_deg_lat) / 1_000_000.0


def ndvi(scene: dict) -> np.ndarray | None:
    b = scene["bands"]
    if "red" in b and "nir" in b:
        r, n = b["red"].astype(np.float64), b["nir"].astype(np.float64)
        return np.clip((n - r) / (n + r + 1e-9), -1, 1)
    return None


def ndwi(scene: dict) -> np.ndarray | None:
    b = scene["bands"]
    if "green" in b and "nir" in b:
        g, n = b["green"].astype(np.float64), b["nir"].astype(np.float64)
        return np.clip((g - n) / (g + n + 1e-9), -1, 1)
    return None


def brightness(scene: dict) -> np.ndarray:
    b = scene["bands"]
    chans = [b[k] for k in ("blue", "green", "red") if k in b]
    src = chans if chans else ([next(iter(b.values()))])
    return np.mean(np.stack(src), axis=0)


def otsu_threshold(gray_u8: np.ndarray) -> float:
    hist, _ = np.histogram(gray_u8, bins=256, range=(0, 256))
    total = gray_u8.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    best_t, best_var = 127.0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, float(t)
    return best_t


def clean_small(mask: np.ndarray, min_px: int = 24) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.uint8)
    labels = _label_components(mask > 0)
    counts = np.bincount(labels.ravel())
    for lab in range(1, counts.size):
        if counts[lab] >= min_px:
            out[labels == lab] = 255
    return out


def _label_components(binary: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage as _ndi

        labeled, _ = _ndi.label(binary)
        return labeled.astype(np.int32)
    except Exception:
        visited = np.zeros_like(binary, dtype=bool)
        labels = np.zeros(binary.shape, dtype=np.int32)
        current = 0
        idx = np.argwhere(binary)
        for y, x in idx:
            if visited[y, x]:
                continue
            current += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                labels[cy, cx] = current
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx_ = cy + dy, cx + dx
                    if 0 <= ny < binary.shape[0] and 0 <= nx_ < binary.shape[1] \
                            and binary[ny, nx_] and not visited[ny, nx_]:
                        visited[ny, nx_] = True
                        stack.append((ny, nx_))
        return labels


def segment_mask(scene: dict, feature: str) -> tuple[np.ndarray | None, str]:
    feature = (feature or "").lower()
    if "water" in feature or "river" in feature or "lake" in feature or not feature:
        n = ndwi(scene)
        if n is not None:
            return clean_small((n > 0.08).astype(np.uint8)), "water"
    if any(k in feature for k in ("veg", "forest", "crop", "agri")):
        n = ndvi(scene)
        if n is not None:
            return clean_small((n > 0.34).astype(np.uint8)), "vegetation"
    if any(k in feature for k in ("urban", "built", "building", "settle")):
        br = brightness(scene).astype(np.float64)
        n = ndvi(scene)
        w = ndwi(scene)
        cand = np.ones_like(br, dtype=bool)
        if n is not None:
            cand &= n < 0.28
        if w is not None:
            cand &= w < 0.06
        thr = otsu_threshold(_percentile_norm(br))
        mask = (cand & (_percentile_norm(br) > thr)).astype(np.uint8)
        return clean_small(mask, min_px=16), "built_up"
    if "cloud" in feature:
        br_u8 = _percentile_norm(brightness(scene))
        return clean_small((br_u8 > 200).astype(np.uint8), min_px=60), "cloud"
    n = ndwi(scene)
    if n is not None:
        return clean_small((n > 0.08).astype(np.uint8)), "water"
    return None, "unknown"


def extract_target(q: str) -> str:
    ql = q.lower()
    order = [
        ("water", ["water", "river", "lake", "reservoir", "pond", "coast"]),
        ("vegetation", ["vegetation", "forest", "crop", "agricultur", "farm", "green"]),
        ("built_up", ["built-up", "built up", "urban", "building", "settlement", "road", "highway"]),
        ("cloud", ["cloud"]),
    ]
    best, pos = "", 10**9
    for target, words in order:
        for wd in words:
            p = ql.find(wd)
            if p != -1 and p < pos:
                best, pos = target, p
    return best


def route_intent(query: str, has_optical: bool, has_sar: bool, has_t2: bool) -> tuple[str, str]:
    q = query.lower()
    fusion_words = ("sar", "fusion", "fuse", "combined", "radar", "backscatter", "cross-modal")
    change_words = ("change", "between t1", "t1 and t2", "before and after", "expanded",
                    "expansion", "temporal", "over time", "difference")
    seg_words = ("highlight", "mask", "segment", "outline", "mark all", "show me all")

    if has_sar and has_optical and any(wd in q for wd in fusion_words):
        return "CROSS_MODAL_FUSION", "Joint SAR+Optical analysis requested; routing to fusion."
    if has_t2 and any(wd in q for wd in change_words):
        tgt = extract_target(q) or "all features"
        return "CHANGE_DETECTION", f"Bi-temporal comparison requested ({tgt}); routing to change detection."
    if any(wd in q for wd in seg_words):
        return "SPATIAL_SEGMENTATION", "Highlight/mask request; routing to segmentation."
    return "VQA_SINGLE", "General scene query; routing to rule-based VQA analysis."


def rgb_render(scene: dict, max_dim: int = 640) -> np.ndarray:
    b = scene["bands"]
    if all(k in b for k in ("blue", "green", "red")):
        chans = [b["blue"], b["green"], b["red"]]
    elif next(iter(b.values())).ndim == 2 and len(b) >= 3:
        keys = list(b.keys())[:3]
        chans = [b[k] for k in keys]
    else:
        g = next(iter(b.values()))
        chans = [g, g, g]
    rgb = np.dstack([_percentile_norm(c.astype(np.float32)) for c in chans])
    h, w = rgb.shape[:2]
    if max(h, w) > max_dim:
        im = Image.fromarray(rgb).resize((w * max_dim // max(h, w), h * max_dim // max(h, w)))
        rgb = np.asarray(im)
    return rgb


def tint(rgb: np.ndarray, mask: np.ndarray | None, color, alpha: float = 0.5) -> bytes:
    canvas = rgb.copy()
    if mask is not None:
        m = mask > 0
        if m.shape[:2] != canvas.shape[:2]:
            mi = Image.fromarray((m * 255).astype(np.uint8)).resize(
                (canvas.shape[1], canvas.shape[0]))
            m = np.asarray(mi) > 0
        for c in range(3):
            ch = canvas[:, :, c].astype(np.float32)
            ch[m] = ch[m] * (1 - alpha) + color[c] * alpha
            canvas[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def mask_stats(mask: np.ndarray, km2: float | None) -> dict:
    px = int(np.count_nonzero(mask))
    total = int(mask.size)
    stats = {
        "pixels": px,
        "percent": round(100.0 * px / max(total, 1), 2),
    }
    if km2 is not None:
        stats["area_km2"] = round(px * km2, 2)
    return stats


def shared_gray(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ga, gb = _percentile_norm(a.astype(np.float32)), _percentile_norm(b.astype(np.float32))
    return ga, gb


def change_analysis(opt: dict, t2: dict) -> dict:
    h = min(opt["shape"][0], t2["shape"][0])
    w = min(opt["shape"][1], t2["shape"][1])

    def gray(sc, hh, ww):
        b = sc["bands"]
        chans = [b[k] for k in ("blue", "green", "red") if k in b][:3]
        if not chans:
            chans = [next(iter(b.values()))]
        g = np.mean([_percentile_norm(c[:hh, :ww].astype(np.float32)) for c in chans], axis=0)

        return g.astype(np.uint8)

    g1, g2 = gray(opt, h, w), gray(t2, h, w)
    diff = np.abs(g1.astype(np.int16) - g2.astype(np.int16)).astype(np.uint8)
    gate_mask = diff >= 4
    gated_vals = diff[gate_mask]
    if gated_vals.size < 50:
        thr = np.inf
        mask = np.zeros_like(diff, dtype=np.uint8)
    else:
        thr = max(otsu_threshold(gated_vals), 10.0)
        mask = (gate_mask & (diff >= thr)).astype(np.uint8)
    changed = int(np.count_nonzero(mask))
    total = int(mask.size)
    delta = (float(g2[mask > 0].mean()) - float(g1[mask > 0].mean())) if changed else 0.0
    direction = "increased" if delta >= 2 else ("decreased" if delta <= -2 else "stable")

    km2 = pixel_km2(opt)
    stats = {
        "change_percent": round(100.0 * changed / max(total, 1), 2),
        "net_direction": direction,
        "threshold": round(float(thr), 1),
    }
    if km2 is not None:
        stats["changed_area_km2"] = round(changed * km2, 2)
    overlay = tint(rgb_render(t2), mask, (255, 40, 40), alpha=0.55)
    return {"stats": stats, "overlay_b64": overlay}


def fusion_analysis(opt: dict, sar: dict) -> dict:
    n = ndwi(opt)
    v = ndvi(opt)
    vv_key = "vv" if "vv" in sar["bands"] else next(iter(sar["bands"]))
    vv_n = _percentile_norm(sar["bands"][vv_key].astype(np.float32)) / 255.0

    water = ((n > 0.05) & (vv_n < 0.38)).astype(np.uint8) if n is not None else None
    built = ((v < 0.30) & (vv_n > 0.55)).astype(np.uint8) if v is not None else None
    veg = (v > 0.38).astype(np.uint8) if v is not None else None

    km2 = pixel_km2(opt)
    stats: dict[str, Any] = {}
    masks = {}
    colors = {"water": (30, 90, 255), "built_up": (255, 120, 30), "vegetation": (40, 190, 70)}
    for name, mk in (("water", water), ("built_up", built), ("vegetation", veg)):
        if mk is None:
            continue
        stats[name] = mask_stats(mk, km2)
        masks[name] = mk
    optical_only_water = int(np.count_nonzero(n > 0.05)) if n is not None else 0
    fused_water = int(np.count_nonzero(water)) if water is not None else 0
    stats["cross_modal"] = {
        "optical_only_water_pixels": optical_only_water,
        "fused_water_pixels": fused_water,
        "sar_rejection_percent": round(100.0 * max(optical_only_water - fused_water, 0)
                                       / max(optical_only_water, 1), 2),
    }

    rgb = rgb_render(opt)
    combined = rgb.copy()
    for name, mk in masks.items():
        m = mk > 0
        if m.shape[:2] != combined.shape[:2]:
            mi = Image.fromarray((m * 255).astype(np.uint8)).resize(
                (combined.shape[1], combined.shape[0]))
            m = np.asarray(mi) > 0
        r_c, g_c, b_c = colors[name]
        for c, cv_ in enumerate((r_c, g_c, b_c)):
            ch = combined[:, :, c].astype(np.float32)
            ch[m] = ch[m] * 0.55 + cv_ * 0.45
            combined[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(combined).save(buf, format="JPEG", quality=82)
    overlay = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"stats": stats, "overlay_b64": overlay}


def vqa_lite(scene: dict) -> str:
    n, v = ndwi(scene), ndvi(scene)
    water_pct = float((n > 0.05).mean() * 100) if n is not None else 0.0
    veg_pct = float((v > 0.38).mean() * 100) if v is not None else 0.0
    other_pct = max(0.0, 100.0 - water_pct - veg_pct)
    dominant = max([("water bodies", water_pct), ("vegetation", veg_pct),
                    ("bare/built surfaces", other_pct)], key=lambda t: t[1])
    return (
        "### Executive Observation\n"
        f"- Scene dominated by **{dominant[0]}** (~{dominant[1]:.1f}% of pixels).\n\n"
        "### Detected Features & Metrics\n"
        f"- Water coverage (NDWI > 0.05): **{water_pct:.2f}%**\n"
        f"- Vegetation (NDVI > 0.38): **{veg_pct:.2f}%**\n"
        f"- Other / built-up candidates: **{other_pct:.2f}%**\n\n"
        "### Confidence & Limitations\n"
        "- Lite serverless engine (rule-based spectral proxies; no SAR texture model).\n"
        "- Full CV engines run on the self-hosted desktop deployment."
    )


def markdown_answer(intent: str, query: str, parts: dict) -> str:
    if intent == "SPATIAL_SEGMENTATION":
        s = parts["stats"]
        area = f", ~{s['area_km2']} km²" if "area_km2" in s else ""
        return (f"### Segmentation — {parts['feature'].title()}\n"
                f"- **{s['pixels']:,} px** mapped ({s['percent']}%{area}).\n"
                f"- Overlay highlights every {parts['feature']} pixel on the optical render.")
    if intent == "CHANGE_DETECTION":
        s = parts["stats"]
        area = f" across ~{s['changed_area_km2']} km²" if "changed_area_km2" in s else ""
        return (f"### Change Detection (T1 → T2)\n"
                f"- Changed pixels: **{s['change_percent']}%**{area}.\n"
                f"- Net signal: **{s['net_direction'].title()}** (Otsu threshold {s['threshold']}).\n"
                f"- Red overlay marks structural change zones.")
    if intent == "CROSS_MODAL_FUSION":
        s, cm = parts["stats"], parts["stats"]["cross_modal"]
        lines = ["### Cross-Modal Fusion (SAR × Optical)\n"]
        for k in ("water", "built_up", "vegetation"):
            if k in s:
                a = f" (~{s[k]['area_km2']} km²)" if "area_km2" in s[k] else ""
                lines.append(f"- **{k.replace('_', ' ').title()}**: {s[k]['percent']}%{a}")
        lines.append(f"- SAR cross-check rejected **{cm['sar_rejection_percent']}%** "
                     "of optical-only water false positives.")
        return "\n".join(lines)
    return vqa_lite(parts["scene"])


_TARGET_RE = re.compile(r"[^a-z0-9_ ]")


def normalize_feature(name: str) -> str:
    return _TARGET_RE.sub("", (name or "").lower()).strip()
