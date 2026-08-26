from __future__ import annotations

import numpy as np

from src.gis.pipeline import (
    RasterData,
    clean_mask,
    colorize_mask,
    compute_ndvi,
    compute_ndwi,
    index_to_mask,
    mask_stats,
    normalize_band,
    overlay_png,
    to_rgb_render,
)

FEATURE_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "water": (30, 90, 255),
    "vegetation": (40, 190, 70),
    "built_up": (255, 120, 30),
    "cloud": (240, 240, 240),
}

FEATURE_ALIASES = {
    "water": ["water", "river", "lake", "reservoir", "pond", "coast"],
    "vegetation": ["vegetation", "forest", "tree", "crop", "agriculture", "agricultural", "farm", "green"],
    "built_up": ["built-up", "built_up", "built up", "urban", "building", "settlement", "infrastructure", "road", "highway"],
    "cloud": ["cloud", "clouds"],
}


def canonical_feature(name: str | None) -> str:
    q = (name or "").lower().strip()
    for canon, aliases in FEATURE_ALIASES.items():
        if any(a in q for a in aliases):
            return canon
    return "water"


def _mask_water(raster: RasterData) -> np.ndarray:
    green, nir = raster.band("green"), raster.band("nir")
    ndwi = compute_ndwi(green.astype(np.float64), nir.astype(np.float64))
    mask = index_to_mask(ndwi, threshold=0.08)
    return clean_mask(mask)


def _mask_vegetation(raster: RasterData) -> np.ndarray:
    red, nir = raster.band("red"), raster.band("nir")
    ndvi = compute_ndvi(red.astype(np.float64), nir.astype(np.float64))
    mask = index_to_mask(ndvi, threshold=0.34)
    return clean_mask(mask)


def _mask_built_up(raster: RasterData) -> np.ndarray:
    red, nir = raster.band("red"), raster.band("nir")
    green = raster.band("green")
    ndvi = compute_ndvi(red.astype(np.float64), nir.astype(np.float64))
    ndwi = compute_ndwi(green.astype(np.float64), nir.astype(np.float64))
    cand = ((ndvi < 0.28) & (ndwi < 0.06)).astype(np.uint8)
    brightness = 0.299 * raster.band("red") + 0.587 * raster.band("green") + 0.114 * nir
    bright_u8 = normalize_band(brightness)
    thr, _ = cv_threshold_otsu(bright_u8)
    texture_like = ((bright_u8 > thr - 12) & (cand > 0)).astype(np.uint8)
    return clean_mask(texture_like, open_ksize=2, min_area_px=16)


def _mask_cloud(raster: RasterData) -> np.ndarray:
    b = [normalize_band(raster.band(n)) for n in ("blue", "green", "red")]
    stack = np.stack(b).astype(np.float32)
    brightness = stack.mean(axis=0)
    saturation = stack.max(axis=0) - stack.min(axis=0)
    mask = ((brightness > 200) & (saturation < 45)).astype(np.uint8)
    return clean_mask(mask, min_area_px=60)


def cv_threshold_otsu(gray_u8: np.ndarray) -> tuple[float, np.ndarray]:
    import cv2

    thr, out = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr), out


def segment_feature(raster: RasterData, target_feature: str | None) -> dict:
    feature = canonical_feature(target_feature)
    dispatch = {
        "water": _mask_water,
        "vegetation": _mask_vegetation,
        "built_up": _mask_built_up,
        "cloud": _mask_cloud,
    }
    mask = dispatch[feature](raster)
    stats = mask_stats(mask, raster.transform, crs=raster.crs)
    rgb = to_rgb_render(raster)
    color = FEATURE_COLORS_RGB[feature]
    overlay = overlay_png(rgb, mask > 0, color=color, alpha=0.5)
    return {
        "feature": feature,
        "mask": mask,
        "stats": stats,
        "overlay_png": overlay,
        "base_rgb": rgb,
        "legend_color": color,
    }
