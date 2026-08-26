@echo off
title SatQuery AI - Deploy to Vercel (public URL)
cd /d E:\SIH

echo ============================================================
echo   SatQuery AI -^> Vercel Deployment
echo ============================================================
echo   Deploys the PUBLIC demo site (lite engine + web client).
echo   Note: the full Streamlit desktop app stays local; this ships
echo   the serverless API + static dashboard.
echo.

where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [0/3] Node.js not found - installing via winget...
    winget install --id OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
    set "PATH=%PATH%;%ProgramFiles%\nodejs"
)

echo [1/3] Signing in to Vercel (browser / device code)...
call npx -y vercel@latest login
if %errorlevel% neq 0 ( echo Login failed. & pause & exit /b 1 )

echo [2/3] Deploying to production...
call npx -y vercel@latest --prod --yes

echo [3/3] Done! Your public URL is printed above (https://your-project.vercel.app)
pause
