from __future__ import annotations

import hashlib

import cv2
import numpy as np
import rasterio
import rasterio.transform
from rasterio.warp import transform as warp_transform

import config
from src.gis.pipeline import RasterData, compute_ndvi, compute_ndwi, pixel_area_km2


def _static_png(key: str, rgb: np.ndarray, jpeg_quality: int = 85) -> str:
    token = hashlib.md5(f"{key}_{rgb.shape}_{jpeg_quality}".encode()).hexdigest()[:12]
    path = config.STATIC_DIR / f"{token}.jpg"
    if not path.exists():
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        path.write_bytes(buf.tobytes())
    return f"app/static/{path.name}"


def center_latlon(raster: RasterData) -> tuple[float, float]:
    h, w = raster.height, raster.width
    lon, lat = rasterio.transform.xy(raster.transform, h // 2, w // 2)
    return float(lat), float(lon)


def render_scene_rgb(raster: RasterData, mask: np.ndarray | None = None,
                     color=(255, 40, 40), alpha: float = 0.5,
                     max_dim: int = 768) -> np.ndarray:
    from src.gis.pipeline import normalize_band, to_rgb_render

    rgb = to_rgb_render(raster)
    if mask is not None:
        m = mask > 0
        for c in range(3):
            ch = rgb[:, :, c].astype(np.float32)
            ch[m] = ch[m] * (1 - alpha) + color[c] * alpha
            rgb[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return rgb


def build_photo_drape_deck(raster: RasterData, scene_key: str = "scene",
                           overlay_mask: np.ndarray | None = None,
                           mask_color=(255, 40, 40), use_terrain: bool = True,
                           pitch: float = 55) -> "object":
    import pydeck as pdk

    rgb = render_scene_rgb(raster, overlay_mask, mask_color)
    texture_url = _static_png(f"drape_{scene_key}", rgb)

    west, south, east, north = _bounds_lonlat(raster)
    lat, lon = center_latlon(raster)

    layers = []
    if use_terrain:
        terrain = pdk.Layer(
            "TerrainLayer",
            elevation_data="https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
            elevation_decoder={"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256,
                               "offset": -32768},
            mesh_max_error=6.0,
        )
        terrain.texture = texture_url
        layers.append(terrain)
    else:
        bitmap = pdk.Layer(
            "BitmapLayer",
            bounds=[west, south, east, north],
            opacity=1.0,
        )
        bitmap.image = texture_url
        layers.append(bitmap)

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=11.5,
                               pitch=pitch, bearing=15)
    return pdk.Deck(layers=layers, initial_view_state=view_state,
                    map_style=None)


def build_class_columns_deck(raster: RasterData, masks: dict[str, np.ndarray],
                             class_heights_m: dict[str, float], exaggeration: float = 6.0,
                             sample_cap: int = 3000) -> "object":
    import pandas as pd
    import pydeck as pdk

    CLASS_COLORS_RGB = {
        "water": [30, 90, 255],
        "vegetation": [40, 190, 70],
        "built_up": [255, 120, 30],
        "cloud": [235, 235, 235],
    }

    rows = []
    per_class_cap = max(400, sample_cap // max(len(masks), 1))
    for name, mask in masks.items():
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        step = max(1, len(ys) // per_class_cap)
        ys, xs = ys[::step], xs[::step]
        lons, lats = rasterio.transform.xy(raster.transform, ys.tolist(), xs.tolist())
        elev = int(round(class_heights_m.get(name, 100.0) * exaggeration))
        r, g, b = CLASS_COLORS_RGB.get(name, [200, 200, 200])
        rows.append(pd.DataFrame({
            "x": [round(v, 6) for v in lons], "y": [round(v, 6) for v in lats],
            "e": elev,
            "r": r, "g": g, "b": b, "c": name[0].upper(),
        }))
    if not rows:
        raise ValueError("No classified pixels to extrude.")
    df = pd.concat(rows, ignore_index=True)

    lat, lon = center_latlon(raster)
    px_m = np.sqrt(pixel_area_km2(raster.transform, crs=raster.crs)) * 1000.0
    radius = max(float(px_m) * 0.62, 8.0)

    layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position=["x", "y"],
        get_elevation="e",
        elevation_scale=1.0,
        radius=radius,
        extruded=True,
        get_fill_color="[r, g, b, 230]",
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(latitude=float(df.y.mean()), longitude=float(df.x.mean()),
                                zoom=11.5, pitch=62, bearing=-18)
    deck = pdk.Deck(
        layers=[layer], initial_view_state=view_state, map_style=None,
        tooltip={"html": "<b>{c}</b>", "style": {"color": "#e2e8f0"}},
    )
    return deck


def _index_grid(raster: RasterData, index: str, step: int) -> tuple[np.ndarray, list[str]]:
    names = [n.lower() for n in raster.band_names]
    arr = raster.array.astype(np.float64)

    def ds(a: np.ndarray) -> np.ndarray:
        z = a[::step, ::step]
        lo, hi = np.percentile(z[np.isfinite(z)], [2, 98])
        return np.clip((z - lo) / max(hi - lo, 1e-9), 0, 1)

    if index == "ndvi" and {"red", "nir"} <= set(names):
        z = compute_ndvi(arr[names.index("red")], arr[names.index("nir")])
        label = "NDVI (-1..1)"
    elif index == "ndwi" and {"green", "nir"} <= set(names):
        z = compute_ndwi(arr[names.index("green")], arr[names.index("nir")])
        label = "NDWI (-1..1)"
    elif index in ("vv", "vh") and index in names:
        z = ds(arr[names.index(index)])
        label = f"SAR {index.upper()} backscatter (norm)"
    else:
        z = ds(np.mean(arr[:3], axis=0))
        label = "Brightness (norm)"
    return z[::1, ::1], label


def plotly_index_surface(raster: RasterData, index: str, step: int = 4,
                         exaggeration: float = 25.0) -> "object":
    import plotly.graph_objects as go

    z, label = _index_grid(raster, index, step)
    zscaled = (z - z.min()) * exaggeration if z.size else z
    colorscale = {
        "ndvi": "RdYlGn", "ndwi": "Bluer",
    }.get(index, "Viridis")

    fig = go.Figure(data=[go.Surface(
        z=zscaled,
        surfacecolor=z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=label, thickness=14, len=0.65),
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15),
    )])
    fig.update_layout(
        title=f"3D Surface — {label}",
        height=680,
        margin=dict(l=0, r=0, t=44, b=0),
        paper_bgcolor="#0b1120",
        font=dict(color="#e2e8f0"),
        scene=dict(
            xaxis_title="pixel E→W", yaxis_title="pixel N→S",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
        ),
    )
    return fig


def plotly_change_surface(t1: RasterData, t2: RasterData, step: int = 4,
                          exaggeration: float = 30.0) -> "object":
    import plotly.graph_objects as go
    from skimage.metrics import structural_similarity

    from src.engines.change_detection import _composite_grays_shared

    g1, g2 = _composite_grays_shared(t1, t2)
    diff = cv2.absdiff(g1, g2).astype(np.float32)
    _, ssim_map = structural_similarity(g1, g2, full=True, data_range=255)
    dissim = ((1 - ssim_map) * 255).astype(np.float32)
    gate = (diff >= 4).astype(np.float32)
    score = np.clip(cv2.addWeighted(diff, 0.55, dissim, 0.45, 0).astype(np.float32) * gate, 0, 255)

    d = score[::step, ::step]
    smooth = cv2.GaussianBlur(d, (5, 5), 0)
    zscaled = smooth * exaggeration / 50.0

    fig = go.Figure(data=[go.Surface(
        z=zscaled,
        surfacecolor=d,
        colorscale=[[0, "#0b1120"], [0.35, "#7f1d1d"], [0.7, "#ef4444"], [1, "#fca5a5"]],
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.85),
    )])
    lat, lon = center_latlon(t1)
    fig.update_layout(
        title=f"Bi-Temporal Change Intensity Field — AOI ({lat:.3f}, {lon:.3f})",
        height=680, margin=dict(l=0, r=0, t=48, b=0),
        paper_bgcolor="#0b1120", font=dict(color="#e2e8f0"),
        scene=dict(aspectmode="cube", camera=dict(eye=dict(x=1.4, y=-1.4, z=0.8))),
    )
    return fig


def _bounds_lonlat(raster: RasterData) -> tuple[float, float, float, float]:
    h, w = raster.height, raster.width
    lons = [raster.transform.c, raster.transform.c + w * raster.transform.a]
    lats = [raster.transform.f, raster.transform.f + h * raster.transform.e]
    try:
        xs, ys = warp_transform(raster.crs, "EPSG:4326", lons, lats)
        west, east = min(xs), max(xs)
        south, north = min(ys), max(ys)
        return west, south, east, north
    except Exception:
        return min(lons), min(lats), max(lons), max(lats)
