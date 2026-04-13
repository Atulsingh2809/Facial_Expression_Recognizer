param(
    [string]$ApiUrl = "http://127.0.0.1:5000",
    [int]$Port = 3000
)

$ErrorActionPreference = "Stop"

Write-Host "=== FER: Run frontend ===" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not (Test-Path (Join-Path $scriptDir "package.json"))) {
    throw "package.json not found in frontend directory: $scriptDir"
}

Write-Host "`n[1/3] Installing frontend dependencies..." -ForegroundColor Yellow
npm install

Write-Host "`n[2/3] Probing API health at $ApiUrl/health ..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$ApiUrl/health" -Method Get -TimeoutSec 8
    Write-Host "API reachable. status=$($health.status), model_loaded=$($health.model_loaded)"
}
catch {
    Write-Host "Warning: API health check failed. Frontend will still start." -ForegroundColor DarkYellow
}

Write-Host "`n[3/3] Starting React dev server..." -ForegroundColor Yellow
$env:REACT_APP_API_URL = $ApiUrl.TrimEnd("/")
$env:PORT = "$Port"
$env:BROWSER = "none"

Write-Host "Frontend: http://localhost:$Port"
Write-Host "API URL:  $env:REACT_APP_API_URL"
npm start
