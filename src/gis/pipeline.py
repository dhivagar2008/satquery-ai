from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
import rasterio

from config import OPTICAL_BANDS, SAR_BANDS

_EPS = 1e-9


@dataclass
class RasterData:
    array: np.ndarray
    transform: object
    crs: object
    band_names: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return self.array.shape[1]

    @property
    def width(self) -> int:
        return self.array.shape[2]

    def band(self, name: str) -> np.ndarray:
        if name not in self.band_names:
            raise ValueError(
                f"Band '{name}' missing. Available bands: {self.band_names}"
            )
        return self.array[self.band_names.index(name)]


def load_geotiff(path: str) -> RasterData:
    try:
        with rasterio.open(path) as src:
            array = src.read().astype(np.float32)
            return RasterData(
                array=array,
                transform=src.transform,
                crs=src.crs,
                band_names=list(src.descriptions) or [],
            )
    except Exception as exc:
        raise ValueError(f"Failed to load GeoTIFF '{path}': {exc}") from exc


def check_crs_match(a: RasterData, b: RasterData) -> None:
    if a.crs is None or b.crs is None:
        raise ValueError("One or both rasters are missing a coordinate system (CRS).")
    if str(a.crs) != str(b.crs):
        raise ValueError(f"CRS mismatch: {a.crs} vs {b.crs}. Re-project before analysis.")


def normalize_band(band: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [low_pct, high_pct])
    if hi <= lo + _EPS:
        return np.zeros(band.shape, dtype=np.uint8)
    scaled = (band - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def to_rgb_render(raster: RasterData, enhance: bool = True) -> np.ndarray:
    arr = raster.array
    if arr.shape[0] >= 3:
        idx = [0, 1, 2]
        names_lower = [n.lower() for n in raster.band_names]
        if "red" in names_lower and "green" in names_lower and "blue" in names_lower:
            idx = [names_lower.index("blue"), names_lower.index("green"), names_lower.index("red")]
        channels = [normalize_band(arr[i]) if enhance else _clip_u8(arr[i]) for i in idx]
        rgb = np.dstack(channels)
    else:
        g = normalize_band(arr[0]) if enhance else _clip_u8(arr[0])
        rgb = np.dstack([g, g, g])

    rgb = cv2.bilateralFilter(rgb, d=5, sigmaColor=25, sigmaSpace=5) if enhance else rgb
    return rgb


def _clip_u8(band: np.ndarray) -> np.ndarray:
    return np.clip(band, 0, 255).astype(np.uint8)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + red + _EPS
    return np.clip((nir - red) / denom, -1.0, 1.0)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = green + nir + _EPS
    return np.clip((green - nir) / denom, -1.0, 1.0)


def compute_ndbi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    denom = swir + nir + _EPS
    return np.clip((swir - nir) / denom, -1.0, 1.0)


def index_to_mask(index_arr: np.ndarray, threshold: float, min_value: float = -1.0,
                  max_value: float = 1.0) -> np.ndarray:
    valid = (index_arr >= min_value) & (index_arr <= max_value)
    mask = (index_arr > threshold) & valid
    return mask.astype(np.uint8)


def clean_mask(mask: np.ndarray, open_ksize: int = 3, close_ksize: int = 5,
               min_area_px: int = 24) -> np.ndarray:
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_o)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_c)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    out = np.zeros_like(cleaned)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            out[labels == i] = 255
    return out


def colorize_mask(mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    m = mask > 0
    out[m] = color[::-1]
    return out


def overlay_png(base_rgb: np.ndarray, mask: np.ndarray | None = None,
                color: tuple[int, int, int] = (255, 40, 40),
                alpha: float = 0.45) -> bytes:
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    if mask is None:
        canvas = base_rgb
    else:
        tinted = base_rgb.copy()
        m = mask > 0
        for c_i, c_val in enumerate(color):
            tinted[m, c_i] = (
                np.clip(tinted[m, c_i].astype(np.float32) * (1 - alpha) + c_val * alpha, 0, 255)
            ).astype(np.uint8)
        canvas = tinted
    ok, buf = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def area_per_pixel_m2(transform, crs=None) -> float:
    det = abs(transform.a * transform.e - transform.b * transform.f)
    if crs is None:
        return det
    try:
        from rasterio.warp import transform as warp_transform

        px_w, px_h = abs(transform.a), abs(transform.e)
        cx, cy = transform.c + px_w / 2.0, transform.f - px_h / 2.0
        xs, _ = warp_transform(crs, "EPSG:3857", [cx - px_w / 2.0, cx + px_w / 2.0], [cy, cy])
        _, ys = warp_transform(crs, "EPSG:3857", [cx, cx], [cy - px_h / 2.0, cy + px_h / 2.0])
        mx = abs(xs[1] - xs[0])
        my = abs(ys[1] - ys[0])
        return max(mx * my, det)
    except Exception:
        return det


def pixel_area_km2(transform, crs=None) -> float:
    return area_per_pixel_m2(transform, crs) / 1_000_000.0


def mask_stats(mask: np.ndarray, transform, crs=None, total_pixels: int | None = None) -> dict:
    px = int(np.count_nonzero(mask))
    total = total_pixels or mask.size
    km2_per_px = pixel_area_km2(transform, crs)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {
        "pixels": px,
        "percent": round(100.0 * px / max(total, 1), 2),
        "area_km2": round(px * km2_per_px, 3),
        "num_regions": len(contours),
    }
