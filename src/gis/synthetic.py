from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_bounds

BBOX_PRESETS = {
    "chennai": (80.15, 12.90, 80.35, 13.10),
    "bengaluru": (77.55, 12.85, 77.75, 13.05),
    "mumbai": (72.80, 18.90, 73.00, 19.10),
    "delhi": (77.10, 28.50, 77.30, 28.70),
    "kolkata": (88.25, 22.50, 88.45, 22.70),
}
BBOX_CHENNAI = BBOX_PRESETS["chennai"]
GRID = 512


def _city_layout(rng: np.random.Generator, grid: int, expansion: float,
                 flip: bool = False) -> tuple[np.ndarray, ...]:
    yy, xx = np.mgrid[0:grid, 0:grid]
    if flip:
        xx = (grid - 1) - xx
    cx, cy = grid // 2, int(grid * 0.42)

    river_mask = np.abs(yy - (0.55 * grid + 18 * np.sin(xx / grid * 6.283))) < 9
    lake_cx, lake_cy, lake_r = int(grid * 0.78), int(grid * 0.72), int(grid * 0.11)
    lake_mask = (xx - lake_cx) ** 2 + (yy - lake_cy) ** 2 < lake_r**2
    water = river_mask | lake_mask

    veg = np.zeros((grid, grid), dtype=bool)
    veg |= (yy > grid * 0.80) & ~water
    veg |= ((xx - int(grid * 0.2)) ** 2 + (yy - int(grid * 0.25)) ** 2) < (int(grid * 0.14)) ** 2
    veg |= rng.random((grid, grid)) < 0.04
    veg &= ~water

    urban_w = int(grid * (0.30 + 0.16 * expansion))
    urban_h = int(grid * (0.22 + 0.20 * expansion))
    urban = (
        (np.abs(xx - cx) < urban_w)
        & (np.abs(yy - cy) < urban_h)
        & ~water
        & ~veg
    )
    road = ((np.abs(yy - cy) < 2) | (np.abs(xx - cx) < 2) |
            (np.abs(yy - int(grid * 0.8)) < 2)) & ~water
    urban |= road

    geom = (xx, yy, cx, cy)
    return water, veg, urban, geom


def evolve_layout(water: np.ndarray, veg: np.ndarray, urban: np.ndarray,
                  geom: tuple, flip: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy, cx, cy = geom
    if flip:
        xx = (xx.shape[1] - 1) - xx
    grid = water.shape[0]
    uw2 = int(grid * 0.30) + int(grid * 0.07)
    uh2 = int(grid * 0.22) + int(grid * 0.08)
    growth_zone = (
        (np.abs(xx - cx) < uw2)
        & (np.abs(yy - cy) < uh2)
        & ~urban
        & ~water
    )
    growth_zone &= ((xx + yy) % 7 != 0)
    urban2 = urban | growth_zone
    veg2 = veg & ~growth_zone

    lake_cx, lake_cy, lake_r = int(grid * 0.78), int(grid * 0.72), int(grid * 0.11)
    r2 = int(lake_r * 0.82)
    lake_shrunk = (xx - lake_cx) ** 2 + (yy - lake_cy) ** 2 < r2**2
    river = np.abs(yy - (0.55 * grid + 18 * np.sin(xx / grid * 6.283))) < 9
    water2 = lake_shrunk | river

    urban2 &= ~water2
    veg2 &= ~water2
    return water2, veg2, urban2


def _optical_bands(water: np.ndarray, veg: np.ndarray, urban: np.ndarray,
                   noise: np.ndarray) -> np.ndarray:
    shape = water.shape
    blue = np.full(shape, 0.06)
    green = np.full(shape, 0.08)
    red = np.full(shape, 0.07)
    nir = np.full(shape, 0.09)

    green[veg] = 0.16; red[veg] = 0.10; nir[veg] = 0.52; blue[veg] = 0.05
    green[urban] = 0.14; red[urban] = 0.15; nir[urban] = 0.20; blue[urban] = 0.13
    green[water] = 0.05; red[water] = 0.03; nir[water] = 0.02; blue[water] = 0.05

    stack = np.clip(np.stack([blue, green, red, nir]) * 10000.0 + noise, 1, 10000).astype(np.uint16)
    return stack


def _sar_bands(water: np.ndarray, veg: np.ndarray, urban: np.ndarray,
               rng: np.random.Generator, grid: int) -> np.ndarray:
    shape = (grid, grid)

    def speckle(mean_db):
        gamma = rng.gamma(shape=4.0, size=shape)
        return mean_db + 10 * np.log10(gamma + 1e-3)

    vv = speckle(-14.0 * np.ones(shape))
    vh = speckle(-19.0 * np.ones(shape))
    vv[veg] += 4.5; vh[veg] += 4.0
    vv[urban] += 11.0; vh[urban] += 8.0
    vv[water] -= 6.0; vh[water] -= 5.0

    stack = np.clip(np.stack([vv, vh]), -40, 10).astype(np.float32)
    return ((stack + 40.0) / 50.0 * 10000.0).astype(np.uint16)


def generate_pair(out_optical_t1: str, out_optical_t2: str, out_sar: str,
                  bbox: tuple[float, float, float, float] = BBOX_CHENNAI,
                  seed: int = 42, grid: int = GRID, flip: bool = False) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, grid, grid)

    w1, v1, u1, geom = _city_layout(rng, grid, expansion=0.0, flip=flip)
    w2, v2, u2 = evolve_layout(w1, v1, u1, geom, flip=flip)

    opt_names = ["blue", "green", "red", "nir"]
    sar_names = ["vv", "vh"]
    profiles_common = dict(driver="GTiff", height=grid, width=grid,
                           transform=transform, crs="EPSG:4326", count=None)

    noise = rng.normal(0, 25, (4, grid, grid))
    paths: dict[str, str] = {}
    for path, layout in [
        (out_optical_t1, (w1, v1, u1)),
        (out_optical_t2, (w2, v2, u2)),
    ]:
        water, veg, urban = layout
        data = _optical_bands(water, veg, urban, noise)
        profile = dict(profiles_common)
        profile["count"] = data.shape[0]
        profile["dtype"] = "uint16"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
            dst.descriptions = tuple(opt_names)
        paths[path] = "optical"

    sar_data = _sar_bands(w2, v2, u2, rng, grid)
    profile = dict(profiles_common)
    profile["count"] = sar_data.shape[0]
    profile["dtype"] = "uint16"
    with rasterio.open(out_sar, "w", **profile) as dst:
        dst.write(sar_data)
        dst.descriptions = tuple(sar_names)
    paths[out_sar] = "sar"

    return {
        "optical_t1": out_optical_t1,
        "optical_t2": out_optical_t2,
        "sar": out_sar,
    }
