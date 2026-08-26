# SatQuery AI 🛰️

**Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries — built for ISRO (Smart India Hackathon SIH26167).**

A full multi-page web application: analysts ask natural-language questions about Sentinel-style satellite imagery (Optical + SAR). An agentic LLM router classifies intent and dispatches to specialist CV engines (VQA, cross-modal fusion, bi-temporal change detection, spatial segmentation), returning structured answers with metrics, overlay PNGs — plus an interactive **3D visualization studio** and a bundled **15-scene dataset**.

## Website Pages (:8501)

| Page | Features |
|---|---|
| 🏠 **Mission Control** | Live stats, feature cards, architecture flow |
| 💬 **Chat Analysis** | Natural-language queries → intent badges → metric cards → overlay viewer with PNG downloads → **router decision-trace JSON** expander → downloadable markdown report |
| 🛰️ **Dataset Gallery** | Thumbnail browser for all scenes, metadata (bands/bbox/CRS/source), one-click "Load into session", multi-file upload import |
| 🌐 **3D Studio** | ① Photo-drape over real DEM terrain (pydeck TerrainLayer) with mask overlays & camera pitch ② NDVI/NDWI/SAR/brightness interactive 3D surfaces (Plotly) ③ Land-cover column extrusions (deck.gl ColumnLayer) ④ Bi-temporal change intensity height-field |
| 📐 **About** | Problem statement, stack table, engine deep-dives, design principles |

## Dataset (`data/raw/`, 15 GeoTIFFs)

Synthetic co-registered Sentinel-style pairs over **Chennai · Bengaluru · Mumbai · Delhi · Kolkata**
— each city ships Optical T1 + T2 (blue/green/red/nir, uint16) + SAR VV/VH, EPSG:4326,
512×512 px ≈ 0.2° (~43 m/px), with a realistic urban-expansion scenario between T1→T2.

```powershell
uv run python scripts/make_synthetic.py                 # regenerate all cities
uv run python scripts/fetch_data.py --aoi chennai      # REAL Sentinel-1/2 via Planetary Computer
```

> Real downloads need an open network; on restricted networks the synthetic dataset powers every feature identically.

## Quick Start

```powershell
# 1. Install uv (if missing)
irm https://astral.sh/uv/install.ps1 | iex

# 2. Create env (Python 3.11 auto-managed) + deps
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv sync

# 3. Generate dataset
uv run python scripts/make_synthetic.py

# 4. Configure Groq (optional — everything falls back offline)
Copy-Item .env.example .env    # then set GROQ_API_KEY

# 5. Launch website + API
uv run streamlit run app.py            # http://localhost:8501
uv run uvicorn main:app --port 8001    # http://localhost:8001/docs
```

## API

`POST /api/query` — multipart: `query`, `optical_file`, `sar_file?`, `bitemporal_file?`
→ JSON: `{intent, reasoning, router_source, tool_pipeline, answer_markdown, metrics[], overlays_b64{}, stats}`

```powershell
uv run python scripts/verify_api.py     # runs all 4 intents against the live server
```

## Demo Queries

| Query | Routed Engine |
|---|---|
| "Describe the land cover in this image" | 🔍 VQA Single |
| "Highlight water bodies" | 🎯 Spatial Segmentation |
| "What changed between T1 and T2?" | ⏱️ Change Detection |
| "Use the optical and SAR images together to identify built-up and water regions" | 🔀 Cross-Modal Fusion |

Every LLM call degrades gracefully to deterministic rule-based CV when offline — **the demo cannot break**.

## Tests

```powershell
uv run pytest tests -v          # 16 engine/router/GIS tests
uv run python scripts/test_pages.py   # headless render of all 5 UI pages
```

## 🔐 Login (Google OAuth)

The site is gated behind a branded login page. Two ways in:

1. **Google sign-in** — configure once:
   - Go to [console.cloud.google.com](https://console.cloud.google.com/apis/credentials) → *Create Credentials → OAuth client ID → Web application*
   - Add authorized redirect URI: `http://localhost:8501/`
   - Copy the client ID/secret into `.env`:
     ```ini
     GOOGLE_CLIENT_ID=xxxx.apps.googleusercontent.com
     GOOGLE_CLIENT_SECRET=xxxx
     ```
   - Restart the app → "Continue with Google" goes live
2. **Guest mode** — one-click demo access, always available (never breaks a live hackathon demo)

## 🚢 Publishing to GitHub (public repo)

Double-click **`PUBLISH_GITHUB.bat`** — it installs the GitHub CLI if needed, walks you through your own secure `gh auth login`, then creates the **public** repo `satquery-ai` and pushes everything.

Manual equivalent:

```powershell
winget install GitHub.cli ; gh auth login
gh repo create satquery-ai --public --source=. --push
```

