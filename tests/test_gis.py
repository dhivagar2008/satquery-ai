import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.gis.pipeline import (
    compute_ndvi,
    compute_ndwi,
    index_to_mask,
    load_geotiff,
    normalize_band,
)
from src.gis.synthetic import generate_pair


@pytest.fixture(scope="module")
def demo_paths(tmp_path_factory):
    out = tmp_path_factory.mktemp("gis")
    t1 = str(out / "opt_t1.tif")
    t2 = str(out / "opt_t2.tif")
    sar = str(out / "sar.tif")
    return generate_pair(t1, t2, sar)


def test_generate_and_load(demo_paths):
    r1 = load_geotiff(demo_paths["optical_t1"])
    assert r1.array.shape == (4, 512, 512)
    assert r1.band_names == ["blue", "green", "red", "nir"]
    sar = load_geotiff(demo_paths["sar"])
    assert sar.array.shape == (2, 512, 512)
    assert sar.crs is not None


def test_missing_band_raises(demo_paths):
    raster = load_geotiff(demo_paths["optical_t1"])
    with pytest.raises(ValueError):
        raster.band("swir")


def test_index_ranges(demo_paths):
    r = load_geotiff(demo_paths["optical_t1"])
    ndvi = compute_ndvi(r.band("red").astype(float), r.band("nir").astype(float))
    ndwi = compute_ndwi(r.band("green").astype(float), r.band("nir").astype(float))
    assert np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0)
    assert np.all(ndwi >= -1.0) and np.all(ndwi <= 1.0)


def test_normalize_output_range():
    band = np.linspace(0, 10000, 65536).reshape(256, 256).astype(np.float32)
    norm = normalize_band(band)
    assert norm.dtype == np.uint8
    assert norm.max() > 200


def test_water_mask_exists_on_demo_scene(demo_paths):
    from src.engines.segmentation import segment_feature

    r = load_geotiff(demo_paths["optical_t1"])
    result = segment_feature(r, "water")
    assert result["stats"]["pixels"] > 1000
    assert 0 < result["stats"]["percent"] < 50
