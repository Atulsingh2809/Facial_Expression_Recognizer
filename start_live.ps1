param(
    [int]$BackendPort = 5000,
    [int]$FrontendPort = 3001
)

$ErrorActionPreference = "Stop"

function Stop-ListeningPort {
    param([int]$Port)
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        try {
            taskkill /PID $c.OwningProcess /F | Out-Host
        } catch {}
    }
}

function Wait-ForHttpOk {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 60
    )
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 400) { return $true }
        } catch {}
        Start-Sleep -Seconds 1
    }
    return $false
}

function Start-CloudflaredQuickTunnel {
    param(
        [Parameter(Mandatory = $true)][string]$OriginUrl,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $cloudflaredExe = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
    if (-not (Test-Path $cloudflaredExe)) {
        throw "cloudflared not found at: $cloudflaredExe. Install with: winget install --id Cloudflare.cloudflared -e"
    }

    $runId = (Get-Date -Format "yyyyMMdd_HHmmss") + "_" + ([guid]::NewGuid().ToString("N"))
    $logPath = Join-Path $env:TEMP ("cloudflared_$Name.$runId.out.log")
    $errPath = Join-Path $env:TEMP ("cloudflared_$Name.$runId.err.log")

    $p = Start-Process -FilePath $cloudflaredExe `
        -ArgumentList @("tunnel", "--url", $OriginUrl) `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $errPath `
        -PassThru

    $regex = [regex]"https://[a-z0-9-]+\.trycloudflare\.com"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $public = $null
    while ($sw.Elapsed.TotalSeconds -lt 90) {
        $txtOut = ""
        $txtErr = ""
        if (Test-Path $logPath) { $txtOut = Get-Content -Path $logPath -Raw -ErrorAction SilentlyContinue }
        if (Test-Path $errPath) { $txtErr = Get-Content -Path $errPath -Raw -ErrorAction SilentlyContinue }
        $txt = ($txtOut + "`n" + $txtErr)
        if ($txt) {
            $m = $regex.Match($txt)
            if ($m.Success) { $public = $m.Value; break }
        }
        Start-Sleep -Milliseconds 500
    }

    if (-not $public) {
        throw "Failed to obtain public URL for $Name tunnel. Check logs: $logPath and $errPath"
    }

    return [pscustomobject]@{
        Name   = $Name
        Origin = $OriginUrl
        Public = $public
        Pid    = $p.Id
        Log    = $logPath
    }
}

function Update-GitHubPagesRedirect {
    param(
        [Parameter(Mandatory = $true)][string]$TargetUrl,
        [Parameter(Mandatory = $true)][string]$RepoDir
    )

    $docsDir = Join-Path $RepoDir "docs"
    $indexPath = Join-Path $docsDir "index.html"
    if (-not (Test-Path $docsDir)) { New-Item -ItemType Directory -Path $docsDir | Out-Null }
    if (-not (Test-Path $indexPath)) { throw "docs/index.html not found at: $indexPath" }

    $html = Get-Content -Path $indexPath -Raw
    # Use simple string replacements to avoid regex/escaping issues in PowerShell.
    $html = $html.Replace("url=https://example.invalid", "url=$TargetUrl")
    $html = $html.Replace('location.replace("https://example.invalid")', "location.replace(`"$TargetUrl`")")
    $html = $html.Replace('<a href="https://example.invalid">https://example.invalid</a>', "<a href=`"$TargetUrl`">$TargetUrl</a>")
    Set-Content -Path $indexPath -Value $html -Encoding UTF8

    Push-Location $RepoDir
    try {
        git add "docs/index.html" | Out-Null
        $status = git diff --cached --name-only
        if ($status) {
            git commit -m "Update live link (GitHub Pages redirect)" | Out-Null
            git push origin main | Out-Null
        }
    }
    finally {
        Pop-Location
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $scriptDir "backend"
$frontendDir = Join-Path $scriptDir "frontend"
$runFrontend = Join-Path $frontendDir "run_frontend.ps1"
$venvPython = "D:\venvs\fer\Scripts\python.exe"

Write-Host "=== FER: Start Live (local + Cloudflare Tunnel) ===" -ForegroundColor Cyan

if (-not (Test-Path $backendDir)) { throw "backend dir not found: $backendDir" }
if (-not (Test-Path $frontendDir)) { throw "frontend dir not found: $frontendDir" }
if (-not (Test-Path $runFrontend)) { throw "run_frontend.ps1 not found: $runFrontend" }
if (-not (Test-Path $venvPython)) { throw "Python venv not found: $venvPython" }

Write-Host "`n[0/5] Stopping anything already listening on ports..." -ForegroundColor Yellow
Stop-ListeningPort -Port $BackendPort
Stop-ListeningPort -Port $FrontendPort

Write-Host "`n[1/5] Starting backend on http://127.0.0.1:$BackendPort ..." -ForegroundColor Yellow
Start-Process -FilePath $venvPython `
    -ArgumentList @("-m", "flask", "--app", "app", "run", "--host", "127.0.0.1", "--port", "$BackendPort", "--no-debugger", "--no-reload") `
    -WorkingDirectory $backendDir | Out-Null

if (-not (Wait-ForHttpOk -Url "http://127.0.0.1:$BackendPort/health" -TimeoutSeconds 90)) {
    throw "Backend did not become healthy on /health"
}

Write-Host "`n[2/5] Creating public backend tunnel..." -ForegroundColor Yellow
$backendTunnel = Start-CloudflaredQuickTunnel -OriginUrl "http://127.0.0.1:$BackendPort" -Name "backend"
Write-Host "Backend tunnel: $($backendTunnel.Public)" -ForegroundColor Green

Write-Host "`n[3/5] Starting frontend on http://127.0.0.1:$FrontendPort (API -> backend tunnel)..." -ForegroundColor Yellow
Start-Process -FilePath "powershell" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runFrontend, "-ApiUrl", $backendTunnel.Public, "-Port", "$FrontendPort") `
    -WorkingDirectory $frontendDir | Out-Null

if (-not (Wait-ForHttpOk -Url "http://127.0.0.1:$FrontendPort" -TimeoutSeconds 120)) {
    throw "Frontend did not become reachable"
}

Write-Host "`n[4/5] Creating public frontend tunnel..." -ForegroundColor Yellow
$frontendTunnel = Start-CloudflaredQuickTunnel -OriginUrl "http://127.0.0.1:$FrontendPort" -Name "frontend"
Write-Host "Frontend tunnel: $($frontendTunnel.Public)" -ForegroundColor Green

Write-Host "`n[5/5] Live links" -ForegroundColor Cyan
Write-Host "APP:     $($frontendTunnel.Public)" -ForegroundColor Green
Write-Host "API:     $($backendTunnel.Public)" -ForegroundColor Green
Write-Host "API /health: $($backendTunnel.Public)/health" -ForegroundColor Green

Write-Host "`n[Pages] Updating GitHub Pages redirect..." -ForegroundColor Yellow
Update-GitHubPagesRedirect -TargetUrl $frontendTunnel.Public -RepoDir $scriptDir
Write-Host "GitHub Pages redirect updated (if git push succeeded)." -ForegroundColor Green

Write-Host "`nKeep this window open to keep the server live." -ForegroundColor Yellow
Write-Host "If you close it, the tunnels stop and links will go offline." -ForegroundColor Yellow

while ($true) { Start-Sleep -Seconds 60 }

