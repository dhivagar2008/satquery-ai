from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import reproject, Resampling

from config import RAW_DIR

AOI_PRESETS = {
    "chennai": (80.15, 12.90, 80.35, 13.10),
    "bengaluru": (77.55, 12.85, 77.75, 13.05),
    "mumbai": (72.80, 18.90, 73.00, 19.10),
}

S2_BANDS = ["B02", "B03", "B04", "B08"]
BAND_NAMES = ["blue", "green", "red", "nir"]


def _bbox(name: str) -> tuple[float, float, float, float]:
    return AOI_PRESETS.get(name.lower(), AOI_PRESETS["chennai"])


def _download_asset(href: str, out_path: Path) -> Path:
    signed = planetary_computer.sign(href)
    with rasterio.open(signed) as src:
        profile = src.profile.copy()
        data = src.read(1)
        transform, width, height = src.transform, src.width, src.height
        crs = src.crs
    profile.update(driver="GTiff", count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
    return out_path


def fetch_pair(aoi: str, date_range_s2: str, date_range_s1: str,
               max_items: int = 1) -> dict[str, str]:
    west, south, east, north = _bbox(aoi)
    bbox = [west, south, east, north]
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    results: dict[str, str] = {}

    s2_search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox,
                               datetime=date_range_s2, query={"eo:cloud_cover": {"lt": 10}})
    s2_items = list(s2_search.items())[:max_items]
    if not s2_items:
        raise RuntimeError("No low-cloud Sentinel-2 items found for AOI/date range.")
    item = s2_items[0]
    band_paths = []
    for band, name in zip(S2_BANDS, BAND_NAMES):
        p = RAW_DIR / f"{aoi}_s2_{name}.tif"
        _download_asset(item.assets[band].href, p)
        band_paths.append(p)

    ref = rasterio.open(band_paths[0])
    stack_path = RAW_DIR / f"{aoi}_optical_stack.tif"
    profile = ref.profile.copy()
    profile.update(count=4, descriptions=tuple(BAND_NAMES))
    with rasterio.open(stack_path, "w", **profile) as dst:
        for i, bp in enumerate(band_paths, start=1):
            with rasterio.open(bp) as bsrc:
                dst.write(bsrc.read(1), i)
    results["optical"] = str(stack_path)

    s1_search = catalog.search(collections=["sentinel-1-grd"], bbox=bbox, datetime=date_range_s1)
    s1_items = list(s1_search.items())
    if s1_items:
        item1 = s1_items[0]
        vv_path = RAW_DIR / f"{aoi}_s1_vv.tif"
        vh_path = RAW_DIR / f"{aoi}_s1_vh.tif"
        _download_asset(item1.assets["vv"].href, vv_path)
        _download_asset(item1.assets["vh"].href, vh_path)

        sar_path = RAW_DIR / f"{aoi}_sar_stack.tif"
        with rasterio.open(vv_path) as vvs, rasterio.open(vh_path) as vhs:
            dst_profile = vvs.profile.copy()
            dst_profile.update(count=2, dtype="float32", descriptions=("vv", "vh"))
            with rasterio.open(sar_path, "w", **dst_profile) as dst:
                data = np.stack([vvs.read(1), vhs.read(1)]).astype(np.float32)
                dst.write(data)
        results["sar"] = str(sar_path)
    else:
        results["sar"] = ""

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch Sentinel-1/2 pair from Planetary Computer")
    parser.add_argument("--aoi", default="chennai", choices=list(AOI_PRESETS.keys()))
    parser.add_argument("--s2-range", default="2024-01-01/2024-03-31")
    parser.add_argument("--s1-range", default="2024-01-01/2024-03-31")
    args = parser.parse_args()
    print(f"Fetching Sentinel-1/2 over {args.aoi} ...")
    res = fetch_pair(args.aoi, args.s2_range, args.s1_range)
    for k, v in res.items():
        print(f"  {k}: {v or 'NOT FOUND'}")


if __name__ == "__main__":
    main()
