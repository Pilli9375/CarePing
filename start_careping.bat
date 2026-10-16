@echo off
title CarePing OS v3.0 QUANTUM - Full-Stack Neural Hospital Launcher
color 0B
cls
echo ===============================================================================
echo    ____               ____  _                ___  ____         _____  ___ 
echo   / ___|__ _ _ __ ___^|  _ \(_)_ __   __ _   / _ \/ ___^|  __   ^|___ / / _ \
echo  ^| ^|   / _` ^| '__/ _ \ ^|_) ^| ^| '_ \ / _` ^| ^| ^| ^| \___ \  \ \ / /^|_ \^| ^| ^| ^|
echo  ^| ^|__^| (_^| ^| ^| ^|  __/  __/^| ^| ^| ^| ^| (_^| ^| ^| ^|_^| ^|___) ^|  \ V /___) ^| ^|_^| ^|
echo   \____\__,_^|_^|  \___^|_^|   ^|_^|_^| ^|_^|\__, ^|  \___/^|____/    \_/^|____/ \___/ 
echo                                     ^|___/                                  
echo   CAREPING OS v3.0 QUANTUM - REACT + NODE.JS + TAILWIND + SCRATCH ML ENGINE
echo   Team AAdharsha Kutumbam: Naga Adithya V (23BCE9062) ^& Harsha Vardhan V
echo ===============================================================================
echo.

cd /d "%~dp0"

echo [STEP 1/3] Running From-Scratch Python ML Training ^& Forensic Audit Pipeline...
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python train_and_evaluate_model.py
) else (
    echo [INFO] Python CLI not in PATH; using pre-computed SVD-OLS ^& IRLS weights.
)
echo.

echo [STEP 2/3] Checking Node.js ^& Express Full-Stack Runtime...
where node >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    if not exist "node_modules" (
        echo [INFO] Installing Node.js + Express + React dependencies ^(first-time setup^)...
        call npm install --no-audit --no-fund
    )
    echo [STEP 3/3] Starting CarePing OS v3.0 Node.js Server on http://localhost:3000 ...
    start "" "http://localhost:3000"
    node server.js
    goto :eof
)

echo [INFO] Node.js not detected in PATH. Launching Python FastAPI/HTTP Server on http://localhost:8050 ...
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    start "" "http://localhost:8050"
    python app.py
    goto :eof
)

echo [STEP 3/3] Opening CarePing OS v3.0 Quantum Standalone Application...
start "" "forgotten_patient_dashboard.html"
pause
