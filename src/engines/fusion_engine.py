from __future__ import annotations

import numpy as np

from src.gis.pipeline import (
    RasterData,
    check_crs_match,
    compute_ndvi,
    compute_ndwi,
    mask_stats,
    normalize_band,
    overlay_png,
    to_rgb_render,
)

CLASS_COLORS_RGB = {
    "water": (30, 90, 255),
    "vegetation": (40, 190, 70),
    "built_up": (255, 120, 30),
}


def _norm_index(band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    a = band_a.astype(np.float64)
    b = band_b.astype(np.float64)
    denom = a + b + 1e-9
    return np.clip((a - b) / denom, -1.0, 1.0)


def _sar_norm(raster: RasterData) -> tuple[np.ndarray, np.ndarray]:
    vv = normalize_band(raster.band("vv")).astype(np.float32) / 255.0
    vh = normalize_band(raster.band("vh")).astype(np.float32) / 255.0
    return vv, vh


def fuse_analysis(optical: RasterData, sar: RasterData, query_type: str | None = None) -> dict:
    try:
        check_crs_match(optical, sar)
    except ValueError:
        pass

    green, nir = optical.band("green"), optical.band("nir")
    red = optical.band("red")
    ndwi = _norm_index(green.astype(np.float64), nir.astype(np.float64))
    ndvi = _norm_index(nir.astype(np.float64), red.astype(np.float64))
    vv, vh = _sar_norm(sar)

    water_mask = ((ndwi > 0.05) & (vv < 0.38)).astype(np.uint8)
    built_mask = ((ndvi < 0.30) & (ndwi < 0.06) & (vv > 0.55)).astype(np.uint8)
    veg_mask = (ndvi > 0.38).astype(np.uint8)

    from src.gis.pipeline import clean_mask

    water_mask = clean_mask(water_mask)
    built_mask = clean_mask(built_mask, open_ksize=2, min_area_px=16)
    veg_mask = clean_mask(veg_mask)

    rgb = to_rgb_render(optical)
    canvas = rgb.copy()
    for name, mask in (("vegetation", veg_mask), ("built_up", built_mask), ("water", water_mask)):
        m = mask > 0
        r_c, g_c, b_c = CLASS_COLORS_RGB[name]
        canvas[m, 0] = np.clip(0.55 * canvas[m, 0] + 0.45 * r_c, 0, 255).astype(np.uint8)
        canvas[m, 1] = np.clip(0.55 * canvas[m, 1] + 0.45 * g_c, 0, 255).astype(np.uint8)
        canvas[m, 2] = np.clip(0.55 * canvas[m, 2] + 0.45 * b_c, 0, 255).astype(np.uint8)

    from src.gis.pipeline import overlay_png as _ov

    fused_png = _ov(canvas, alpha=1.0)

    stats = {}
    for name, mask in (("water", water_mask), ("built_up", built_mask), ("vegetation", veg_mask)):
        stats[name] = mask_stats(mask, optical.transform, crs=optical.crs)

    optical_only_water = int(np.count_nonzero(ndwi > 0.05))
    fused_water_px = int(np.count_nonzero(water_mask))
    sar_rejected = max(optical_only_water - fused_water_px, 0)
    stats["cross_modal"] = {
        "optical_only_water_pixels": optical_only_water,
        "fused_water_pixels": fused_water_px,
        "sar_rejected_false_positives": sar_rejected,
        "sar_rejection_percent": round(
            100.0 * sar_rejected / max(optical_only_water, 1), 2
        ),
        "query_type": query_type or "land_cover",
    }

    return {
        "stats": stats,
        "masks": {"water": water_mask, "built_up": built_mask, "vegetation": veg_mask},
        "overlay_png": fused_png,
        "base_rgb": rgb,
    }
