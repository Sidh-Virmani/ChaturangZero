$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$uiDir = Join-Path $projectRoot 'shared/ui'
$venvPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error 'Node.js and npm are required to start ChaturangZero.'
    exit 1
}
if (-not (Test-Path -LiteralPath (Join-Path $uiDir 'package.json'))) {
    Write-Error "Frontend package not found in $uiDir."
    exit 1
}
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    Write-Error 'Python is required to start ChaturangZero.'
    exit 1
}

if (-not (Test-Path -LiteralPath (Join-Path $uiDir 'node_modules'))) {
    Push-Location $uiDir
    try {
        npm install
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        Pop-Location
    }
}

Write-Host 'Starting ChaturangZero at http://127.0.0.1:5173/'
Write-Host 'Press Ctrl+C to stop the application.'
& $python (Join-Path $projectRoot 'run_chaturangzero.py')
exit $LASTEXITCODE
