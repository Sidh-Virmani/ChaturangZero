@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "UI_DIR=%PROJECT_ROOT%shared\ui"

where npm >nul 2>&1
if errorlevel 1 (
  echo Node.js and npm are required to start ChaturangZero.
  echo Install Node.js, then run this launcher again.
  pause
  exit /b 1
)

if not exist "%UI_DIR%\package.json" (
  echo Frontend package not found at "%UI_DIR%".
  pause
  exit /b 1
)

pushd "%UI_DIR%"

if not exist "node_modules\" (
  echo Installing frontend dependencies...
  call npm install
  if errorlevel 1 goto :failed
)

echo Starting ChaturangZero at http://localhost:43147/
echo Close this window or press Ctrl+C to stop the application.

rem Future Python backend startup can be added here before the frontend process.
if defined CHATURANGZERO_NO_OPEN (
  call npm run dev -- --host 127.0.0.1 --port 43147 --strictPort
) else (
  call npm run dev -- --host 127.0.0.1 --port 43147 --strictPort --open
)
if errorlevel 1 goto :failed

popd
exit /b 0

:failed
echo.
echo ChaturangZero could not be started.
popd
pause
exit /b 1
