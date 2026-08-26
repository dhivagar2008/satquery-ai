from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RAW_DIR
from src.gis.synthetic import BBOX_PRESETS, generate_pair


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic co-registered S1/S2 GeoTIFF demo scenes")
    parser.add_argument("--aoi", default="all", choices=list(BBOX_PRESETS.keys()) + ["all"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grid", type=int, default=512)
    args = parser.parse_args()

    aois = list(BBOX_PRESETS.keys()) if args.aoi == "all" else [args.aoi]
    for i, aoi in enumerate(aois):
        seed = args.seed if args.seed is not None else 42 + i * 7
        t1 = str(RAW_DIR / f"{aoi}_optical_t1.tif")
        t2 = str(RAW_DIR / f"{aoi}_optical_t2.tif")
        sar = str(RAW_DIR / f"{aoi}_sar.tif")
        paths = generate_pair(t1, t2, sar, bbox=BBOX_PRESETS[aoi],
                              seed=seed, grid=args.grid, flip=(i % 2 == 1))
        print(f"[{aoi}] seed={seed}")
        for k, v in paths.items():
            print(f"   {k}: {Path(v).name}")


if __name__ == "__main__":
    main()
