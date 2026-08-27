# SatQuery AI – Deployment Checklist

## Pre-flight
- [x] `.env` secrets cleared from repo; `.env.example` committed
- [x] `.gitignore` excludes `.env*` but keeps `!.env.example`
- [x] `.vercelignore` ships only `api/`, `public/`, `vercel.json`, `api/requirements.txt`
- [x] `vercel.json` builds `@vercel/python` for `api/index.py` with `includeFiles: public/**`
- [x] `api/requirements.txt` minimal: fastapi, uvicorn, python-multipart, numpy, pillow, tifffile, scipy
- [x] DeepSeek support merged into `config.py` and `src/orchestrator/router.py`

## Vercel Public Demo
1. Push to GitHub
2. New Vercel project → import repo → Framework = Other
3. Env vars (optional): `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`
4. Deploy. Verify:
   - `GET /api/health` → `{status:"ok",engine:"lite-serverless"}`
   - `POST /api/query` with optical GeoTIFF returns `intent`, `metrics`, `overlays_b64`

## Streamlit Full App
- Desktop: `uv run streamlit run app.py`
- Cloud: Streamlit Community Cloud, app path `app.py`, requirements `requirements.txt`, `packages.txt`
- First run auto-generates synthetic dataset via `src/bootstrap.ensure_dataset()`

## Notes
- Lite engine is rule-based and works offline. LLM routing is optional.
- Data files in `data/raw/` are git-ignored; synthetic data can be regenerated with `scripts/make_synthetic.py`
