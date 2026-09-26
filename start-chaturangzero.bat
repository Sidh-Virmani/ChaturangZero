@echo off
setlocal
cd /d "%~dp0"
set "CZ_OPEN=--open"
if defined CHATURANGZERO_NO_OPEN set "CZ_OPEN="
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run.py --setup %CZ_OPEN%
) else (
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3 run.py --setup %CZ_OPEN%
  ) else (
    python run.py --setup %CZ_OPEN%
  )
)
if errorlevel 1 (
  echo.
  echo See the message above. Python 3.12 and Node.js 22.12+ are recommended.
  pause
  exit /b 1
)
