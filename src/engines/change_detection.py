from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from src.gis.pipeline import (
    RasterData,
    check_crs_match,
    mask_stats,
    overlay_png,
    pixel_area_km2,
    to_rgb_render,
)


def _composite_grays_shared(raster_a: RasterData, raster_b: RasterData) -> tuple[np.ndarray, np.ndarray]:
    n_bands = min(3, raster_a.array.shape[0], raster_b.array.shape[0])
    grays = []
    for i in range(n_bands):
        ba, bb = raster_a.array[i].astype(np.float32), raster_b.array[i].astype(np.float32)
        finite = np.concatenate([ba[np.isfinite(ba)], bb[np.isfinite(bb)]])
        if finite.size == 0:
            grays.append((np.zeros(ba.shape, np.uint8), np.zeros(bb.shape, np.uint8)))
            continue
        lo, hi = np.percentile(finite, [2.0, 98.0])
        if hi <= lo + 1e-9:
            grays.append((np.zeros(ba.shape, np.uint8), np.zeros(bb.shape, np.uint8)))
            continue
        ga = np.clip((ba - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        gb = np.clip((bb - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        grays.append((ga, gb))
    g_a = np.mean(np.stack([g[0] for g in grays]), axis=0).astype(np.uint8)
    g_b = np.mean(np.stack([g[1] for g in grays]), axis=0).astype(np.uint8)
    return g_a, g_b


def detect_changes(t1: RasterData, t2: RasterData, target_class: str | None = None,
                   sensitivity: float = 1.0) -> dict:
    if t1.array.shape[1:] != t2.array.shape[1:]:
        raise ValueError(
            f"Bi-temporal images must share dimensions: {t1.array.shape[1:]} vs {t2.array.shape[1:]}"
        )
    try:
        check_crs_match(t1, t2)
    except ValueError:
        pass

    g1, g2 = _composite_grays_shared(t1, t2)

    diff = cv2.absdiff(g1, g2)
    _, ssim_map = structural_similarity(g1, g2, full=True, data_range=255)
    dissim = ((1.0 - ssim_map) * 255).astype(np.uint8)

    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
    dissim_blur = cv2.GaussianBlur(dissim, (3, 3), 0)

    gate = (diff >= 4).astype(np.float32)
    score = (cv2.addWeighted(diff_blur, 0.55, dissim_blur, 0.45, 0).astype(np.float32) * gate)
    score = np.clip(score, 0, 255).astype(np.uint8)
    if sensitivity != 1.0:
        score = np.clip(score.astype(np.float32) * sensitivity, 0, 255).astype(np.uint8)

    thr, change_mask_u8 = cv2.threshold(
        score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    change_mask_u8 = cv2.morphologyEx(change_mask_u8, cv2.MORPH_OPEN, kernel)
    change_mask_u8 = cv2.morphologyEx(change_mask_u8, cv2.MORPH_CLOSE, kernel)
    num, labels, stats_cc, _ = cv2.connectedComponentsWithStats(change_mask_u8, connectivity=8)
    refined = np.zeros_like(change_mask_u8)
    for i in range(1, num):
        if stats_cc[i, cv2.CC_STAT_AREA] >= 20:
            refined[labels == i] = 255

    changed_px = int(np.count_nonzero(refined))
    km2_per_px = pixel_area_km2(t1.transform, crs=t1.crs)
    total_px = refined.size
    delta_inside = (
        g2.astype(np.float32)[refined > 0].mean() - g1.astype(np.float32)[refined > 0].mean()
        if changed_px > 0 else 0.0
    )
    net_direction = "stable"
    if abs(delta_inside) >= 2.0:
        net_direction = "increased" if delta_inside > 0 else "decreased"

    rgb_t2 = to_rgb_render(t2)
    overlay = overlay_png(rgb_t2, refined > 0, color=(255, 40, 40), alpha=0.55)

    return {
        "mask": refined,
        "threshold": round(float(thr), 2),
        "stats": {
            "changed_pixels": changed_px,
            "change_percent": round(100.0 * changed_px / max(total_px, 1), 2),
            "changed_area_km2": round(changed_px * km2_per_px, 3),
            "total_area_km2": round(total_px * km2_per_px, 3),
            "num_change_regions": int(num - 1),
            "net_direction": net_direction,
            "target_class": target_class or "all_features",
        },
        "overlay_png": overlay,
        "base_rgb": rgb_t2,
    }
