from __future__ import annotations

"""First-run dataset bootstrap for cloud deployments.

The synthetic GeoTIFF demo scenes are git-ignored (too large for the repo),
so a fresh cloud clone has an empty data/raw directory. This module
regenerates the bundled 15-scene dataset (5 cities x optical T1/T2 + SAR)
on first launch using the exact same generator as scripts/make_synthetic.py.
"""

from src.gis.synthetic import BBOX_PRESETS, generate_pair
from config import RAW_DIR


def raw_scene_count() -> int:
    return len(list(RAW_DIR.glob("*.tif")))


def ensure_dataset(force: bool = False) -> int:
    """Generate all synthetic demo scenes if missing. Returns scene count."""
    if not force and raw_scene_count() > 0:
        return raw_scene_count()

    count = 0
    for i, aoi in enumerate(BBOX_PRESETS):
        seed = 42 + i * 7
        generate_pair(
            str(RAW_DIR / f"{aoi}_optical_t1.tif"),
            str(RAW_DIR / f"{aoi}_optical_t2.tif"),
            str(RAW_DIR / f"{aoi}_sar.tif"),
            bbox=BBOX_PRESETS[aoi],
            seed=seed,
            grid=512,
            flip=(i % 2 == 1),
        )
        count += 3
    return count
