from __future__ import annotations

import base64
import cv2
import numpy as np

import config
from src.gis.pipeline import (
    RasterData,
    compute_ndvi,
    compute_ndwi,
    normalize_band,
    to_rgb_render,
)

VQA_SYSTEM_PROMPT = """You are an expert Remote Sensing Scientist and Computer Vision AI developed for the Indian Space Research Organisation (ISRO).
You are analyzing co-registered Sentinel-1 SAR and Sentinel-2 Multispectral satellite imagery.

GUIDELINES FOR ANALYSIS:
1. Distinguish between Optical signatures (RGB, Near-Infrared vegetation response) and SAR backscatter (surface roughness, dielectric properties, water specular reflection).
2. For land-cover classification, cite spectral indices (NDVI for vegetation, NDWI for water, NDBI for built-up) where applicable.
3. Be precise with spatial scale, texture, and pattern identification. Avoid speculative assumptions beyond the provided image resolution.
4. Structure your response clearly into:
   - Executive Observation
   - Detected Features & Land Cover Metrics
   - Confidence Assessment & Potential Sensor Limitations (e.g., cloud occlusion in optical, layover in SAR)

ANSWER FORMAT: Structured Markdown with bullet points. Be concise."""


def _png_data_url(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def rule_based_answer(raster: RasterData, user_query: str) -> tuple[str, dict]:
    rgb = to_rgb_render(raster)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)

    green = raster.band("green").astype(np.float64)
    nir = raster.band("nir").astype(np.float64)
    red = raster.band("red").astype(np.float64)
    ndwi = compute_ndwi(green, nir)
    ndvi = compute_ndvi(nir, red)

    water_pct = float((ndwi > 0.05).mean() * 100)
    veg_pct = float((ndvi > 0.38).mean() * 100)
    built_pct = float(((ndvi <= 0.30) & (ndwi <= 0.06)).mean() * 100)
    brightness = float(gray.mean())

    metrics = {
        "water_pct": round(water_pct, 2),
        "vegetation_pct": round(veg_pct, 2),
        "built_up_candidate_pct": round(built_pct, 2),
        "edge_density": round(edge_density, 4),
        "mean_brightness": round(brightness, 1),
        "ndvi_mean": round(float(ndvi.mean()), 3),
        "ndwi_mean": round(float(ndwi.mean()), 3),
    }

    dominant = max(
        [("water bodies", water_pct), ("vegetation / cropland", veg_pct), ("built-up surfaces", built_pct)],
        key=lambda t: t[1],
    )[0]
    structure_note = (
        "high structural complexity with dense linear features (roads/blocks)"
        if edge_density > 0.06 else "moderate structural texture"
    )
    confidence = "medium" if brightness > 25 else "low (dark/low-signal scene)"

    answer = (
        "### Executive Observation\n"
        f"- Scene is dominated by **{dominant}** "
        f"(~{max(metrics['water_pct'], metrics['vegetation_pct'], metrics['built_up_candidate_pct']):.1f}% of pixels).\n"
        f"- Texture analysis indicates {structure_note} (edge density {edge_density:.3f}).\n\n"
        "### Detected Features & Land Cover Metrics\n"
        f"| Class | Coverage |\n|---|---|\n"
        f"| Water (NDWI > 0.05) | {metrics['water_pct']}% |\n"
        f"| Vegetation (NDVI > 0.38) | {metrics['vegetation_pct']}% |\n"
        f"| Built-up candidates | {metrics['built_up_candidate_pct']}% |\n\n"
        "### Confidence Assessment & Sensor Limitations\n"
        f"- Rule-based CV analysis (VLM offline); confidence: **{confidence}**.\n"
        "- Spectral proxies only — no SWIR/CIR bands available; SAR texture not fused in this answer."
    )
    return answer, metrics


def answer_query(rasters: list[RasterData], user_query: str) -> tuple[str, dict, str]:
    primary = rasters[0]
    rgb = to_rgb_render(primary)

    if config.llm_available():
        try:
            from groq import Groq

            client = Groq(api_key=config.GROQ_API_KEY)
            content: list[dict] = [
                {"type": "text",
                 "text": f"QUERY: {user_query}\nCONTEXT: Sentinel-2 optical RGB+NIR imagery"
                         + (" plus co-registered Sentinel-1 SAR VV/VH." if len(rasters) > 1 else ".")}
            ]
            for r in rasters[:2]:
                content.append({"type": "image_url", "image_url": {"url": _png_data_url(to_rgb_render(r))}})
            completion = client.chat.completions.create(
                model=config.GROQ_VISION_MODEL,
                temperature=0.2,
                max_tokens=900,
                messages=[
                    {"role": "system", "content": VQA_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
            text = completion.choices[0].message.content or ""
            if text.strip():
                return text.strip(), {}, "groq_vision"
        except Exception:
            pass

    answer, metrics = rule_based_answer(primary, user_query)
    return answer, metrics, "rule_based_cv"
