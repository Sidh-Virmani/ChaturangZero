@echo off
title ChaturangZero — Start

echo.
echo  ============================================
echo    ChaturangZero — Starting...
echo  ============================================
echo.

REM ── Check Python ──────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause & exit /b 1
)

REM ── Check Node ────────────────────────────────────────────────────────────
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    pause & exit /b 1
)

REM ── Install Python deps (first time only) ─────────────────────────────────
if not exist ".venv" (
    echo [setup] Creating Python virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [setup] Installing Python dependencies...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

REM ── Install UI deps (first time only) ─────────────────────────────────────
if not exist "shared\ui\node_modules" (
    echo [setup] Installing Node dependencies...
    cd shared\ui
    npm install
    cd ..\..
)

echo.
echo  Starting the Python backend (FastAPI on port 8000)...
start "ChaturangZero Backend" cmd /k "call .venv\Scripts\activate.bat && uvicorn integration.api:app --reload --port 8000"

echo  Waiting 3 seconds for backend to start...
timeout /t 3 /nobreak >nul

echo  Starting the React UI (Vite on port 5173)...
start "ChaturangZero UI" cmd /k "cd shared\ui && npm run dev"

echo.
echo  ============================================
echo    Open in your browser:
echo      UI:      http://localhost:5173
echo      API:     http://localhost:8000
echo      API docs: http://localhost:8000/docs
echo  ============================================
echo.
echo  Press any key to close this window.
echo  (The backend and UI will keep running.)
pause >nul
