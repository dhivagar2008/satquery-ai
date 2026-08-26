@echo off
title SatQuery AI - Publish to GitHub (public repo: satquery-ai)
cd /d E:\SIH

echo ============================================================
echo   SatQuery AI -^> GitHub Publisher  (public repo: satquery-ai)
echo ============================================================
echo   This will:
echo     1. Install GitHub CLI if missing
echo     2. Sign you in to GitHub (browser/device code - YOUR login)
echo     3. Create PUBLIC repo "satquery-ai" and push this project
echo.

where gh >nul 2>nul
if %errorlevel% neq 0 (
    echo [1/3] GitHub CLI not found - installing via winget...
    winget install --id GitHub.cli --accept-source-agreements --accept-package-agreements
    if %errorlevel% neq 0 (
        echo   winget failed - trying direct installer...
        curl -L -o "%TEMP%\gh_setup.msi" https://github.com/cli/cli/releases/latest/download/gh_windows_amd64.msi
        msiexec /i "%TEMP%\gh_setup.msi" /qn
    )
    echo   Refreshing PATH...
    set "PATH=%PATH%;%ProgramFiles%\GitHub CLI"
)

where gh >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: GitHub CLI still not available. Install manually: https://cli.github.com
    pause & exit /b 1
)

echo [2/3] Checking GitHub sign-in...
gh auth status >nul 2>nul
if %errorlevel% neq 0 (
    gh auth login --hostname github.com --git-protocol https --web
)

echo [3/3] Creating PUBLIC repo satquery-ai and pushing...
gh repo create satquery-ai --public --source=. --remote=origin --push --description "SatQuery AI - Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis (ISRO / SIH26167)"

if %errorlevel% equ 0 (
    echo.
    echo   SUCCESS! Your public repository:
    echo   https://github.com/%USERNAME%/satquery-ai
) else (
    echo   Repo creation returned an error - if it says already exists, run:
    echo     git remote add origin https://github.com/^<your-user^>/satquery-ai.git ^&^& git push -u origin master
)
pause
