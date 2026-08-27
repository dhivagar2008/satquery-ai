import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile").strip()
GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
).strip()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/deepseek-v4-pro-0813").strip()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI", "http://localhost:8501/"
).strip()
GOOGLE_OAUTH_ENABLED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

APP_AUTHOR = "Dhivagar R"
APP_TAGLINE = "SatQuery AI — SIH26167"

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OVERLAY_DIR = DATA_DIR / "overlays"
STATIC_DIR = BASE_DIR / "static"
THUMB_DIR = STATIC_DIR / "thumbs"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, OVERLAY_DIR, STATIC_DIR, THUMB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

OPTICAL_BANDS = ["blue", "green", "red", "nir"]
SAR_BANDS = ["vv", "vh"]


def llm_available() -> bool:
    return bool(GROQ_API_KEY or DEEPSEEK_API_KEY)


def ensure_importable():
    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
