@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "UI_DIR=%PROJECT_ROOT%shared\ui"

where npm >nul 2>&1
if errorlevel 1 (
  echo Node.js and npm are required to start ChaturangZero.
  echo Install Node.js, then run this launcher again.
  exit /b 1
)

if not exist "%UI_DIR%\package.json" (
  echo Frontend package not found at "%UI_DIR%".
  exit /b 1
)

set "PYTHON_CMD=python"
if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" set "PYTHON_CMD=%PROJECT_ROOT%.venv\Scripts\python.exe"
if "%PYTHON_CMD%"=="python" (
  where python >nul 2>&1
  if errorlevel 1 (
    echo Python is required to run ChaturangZero.
    exit /b 1
  )
)

pushd "%UI_DIR%"

if not exist "node_modules\" (
  echo Installing frontend dependencies...
  call npm install
  if errorlevel 1 (
    popd
    goto :failed
  )
)

echo Starting ChaturangZero at http://127.0.0.1:5173/
echo Close this window or press Ctrl+C to stop the application.

popd
"%PYTHON_CMD%" "%PROJECT_ROOT%run_chaturangzero.py"
exit /b %errorlevel%
