param(
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"

Write-Host "=== FER: Train model + run API ===" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelDir = Join-Path $scriptDir "model"
$trainScript = Join-Path $modelDir "train.py"
$modelPath = Join-Path $modelDir "fer_model.h5"

if (-not (Test-Path $trainScript)) {
    throw "train.py not found at: $trainScript"
}

if (-not (Test-Path $CsvPath)) {
    throw "FER2013 CSV not found: $CsvPath"
}

Write-Host "`n[1/3] Installing backend dependencies..." -ForegroundColor Yellow
Set-Location $scriptDir
pip install -r requirements.txt

Write-Host "`n[2/3] Training model..." -ForegroundColor Yellow
Set-Location $modelDir
python train.py --csv $CsvPath --out-dir .

if (-not (Test-Path $modelPath)) {
    throw "Training finished but model file missing: $modelPath"
}

Write-Host "`n[3/3] Starting Flask API..." -ForegroundColor Yellow
Write-Host "Model: $modelPath"
Write-Host "API:   http://$BindHost`:$Port"

Set-Location $scriptDir
$env:FLASK_APP = "app.py"
python -m flask run --host $BindHost --port $Port
