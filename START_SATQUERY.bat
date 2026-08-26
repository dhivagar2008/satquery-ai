@echo off
title SatQuery AI Launcher
cd /d E:\SIH

echo ============================================
echo   SatQuery AI - starting services
echo ============================================

start "SatQuery API (:8001)" cmd /k ".venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8001"

start "SatQuery Web (:8501)" cmd /k ".venv\Scripts\streamlit.exe run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501 --server.fileWatcherType none"

timeout /t 15 /nobreak >nul
start "" "http://localhost:8501"

echo.
echo   Website : http://localhost:8501
echo   API     : http://localhost:8001/docs
echo.
echo   Two service windows opened - keep them open while using the site.
echo   Close those windows to stop SatQuery AI.
echo.
pause
