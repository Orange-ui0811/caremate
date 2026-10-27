param([switch]$PrepareOnly, [switch]$StartToolsOnly, [switch]$StartClassifierOnly, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $root
$env:UV_CACHE_DIR = Join-Path $root 'data\runtime-cache\uv'
$env:HF_HOME = Join-Path $root 'data\runtime-cache\huggingface'
$envFile = Join-Path $root '.env'
$python = Join-Path $root '.venv\Scripts\python.exe'
$logDir = Join-Path $root 'log'
$stateDir = Join-Path $root 'data\windows-launcher'
New-Item -ItemType Directory -Force -Path $logDir,$stateDir | Out-Null

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE). See the output above." }
}
function Test-Port([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try { $client.Connect('127.0.0.1', $Port); return $true }
    catch { return $false }
    finally { $client.Dispose() }
}
function Get-OwnedProcess([string]$Name) {
    $file = Join-Path $stateDir "$Name.json"
    if (!(Test-Path -LiteralPath $file)) { return $null }
    $saved = Get-Content -LiteralPath $file -Raw | ConvertFrom-Json
    $process = Get-Process -Id $saved.Id -ErrorAction SilentlyContinue
    $processPath = if ($process) { $process.Path } else { $null }
    # 32-bit PowerShell (launched by GNU Make) cannot read a 64-bit process MainModule.
    if ($process -and !$processPath) {
        $processPath = (Get-CimInstance Win32_Process -Filter "ProcessId = $($saved.Id)" -ErrorAction Stop).ExecutablePath
    }
    if ($process -and $process.StartTime.ToUniversalTime().Ticks.ToString() -eq $saved.StartTicks -and $processPath -eq $python) {
        return $process
    }
    return $null
}
function Start-ServiceProcess([string]$Name, [int]$Port, [string]$Arguments) {
    $existing = Get-OwnedProcess $Name
    if ($existing) {
        if (!(Test-Port $Port)) { throw "$Name is running but port $Port is unavailable. Use Stop-CareMate.cmd and retry." }
        Write-Host "$Name is already running on port $Port."
        return
    }
    if (Test-Port $Port) { throw "Port $Port is occupied by another process. It has not been stopped." }
    $process = Start-Process -FilePath $python -ArgumentList $Arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "$Name.stdout.log") -RedirectStandardError (Join-Path $logDir "$Name.stderr.log")
    @{ Id = $process.Id; StartTicks = $process.StartTime.ToUniversalTime().Ticks.ToString() } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stateDir "$Name.json") -Encoding UTF8
    $script:startedNames.Add($Name)
    $deadline = (Get-Date).AddSeconds(90)
    do {
        $process.Refresh()
        if ($process.HasExited) { throw "$Name exited. See log\$Name.stderr.log." }
        if (Test-Port $Port) { Write-Host "$Name ready on port $Port."; return }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    throw "$Name did not become ready. See log\$Name.stderr.log."
}
$script:startedNames = New-Object 'System.Collections.Generic.List[string]'
try {
    if ($StartClassifierOnly) {
        if (!(Test-Path -LiteralPath (Join-Path $root 'data\ch10\onnx\model.onnx'))) {
            throw 'Classifier model is missing. Complete ch10-train and ch10-export first.'
        }
        Start-ServiceProcess 'topic-classifier' 8110 '-m scripts.ch10.serve'
        exit 0
    }
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (!$uv -and (Test-Path 'D:\Hermes\bin\uv.exe')) { $uv = Get-Command 'D:\Hermes\bin\uv.exe' }
    if (!$uv) { throw 'uv is missing. Install uv first.' }
    if (!(Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker is missing. Install Docker Desktop first.' }
    if (!(Test-Path -LiteralPath $envFile)) { throw 'Missing .env. Copy .env.example and fill in your model API keys.' }

    $values = @{}
    foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
        if ($line -match '^\s*([A-Z_]+)\s*=\s*(.*?)\s*$') { $values[$matches[1]] = $matches[2].Trim('"',"'") }
    }

    Write-Host '[1/5] Preparing Python dependencies...'
    Invoke-Checked $uv.Source @('sync','--locked','--python','3.13','--group','ml')
    Write-Host '[2/5] Starting database and search services...'
    Invoke-Checked 'docker' @('info','--format','{{.OSType}}')
    Invoke-Checked 'docker' @('compose','up','-d','--wait','--wait-timeout','300')
    Invoke-Checked 'docker' @('compose','exec','-T','mysql','mysql','--default-character-set=utf8mb4','-uroot','-proot','mewhelp','-e','SELECT COUNT(*) AS faq_rows FROM faq;')
    if ($values['LANGFUSE_PUBLIC_KEY'] -and $values['LANGFUSE_SECRET_KEY']) {
        Write-Host 'Starting local observability services...'
        Invoke-Checked 'docker' @('compose','-f','docker-compose.langfuse.yml','up','-d','--wait','--wait-timeout','300')
        $deadline = (Get-Date).AddMinutes(5)
        do {
            try {
                $health = Invoke-RestMethod 'http://127.0.0.1:3000/api/public/health' -TimeoutSec 10
                if ($health.status -eq 'OK') { break }
            } catch { }
            Start-Sleep -Seconds 3
        } while ((Get-Date) -lt $deadline)
        if (!$health -or $health.status -ne 'OK') { throw 'Langfuse did not become ready on port 3000.' }
    }
    if ($PrepareOnly) { Write-Host 'Dependencies and containers are ready. Fill .env, then double-click Start-CareMate.cmd.'; exit 0 }
    if ($StartToolsOnly) {
        Start-ServiceProcess 'mcp-logistics' 8101 '-m mcp_servers.logistics_server'
        Start-ServiceProcess 'mcp-aftersales' 8102 '-m mcp_servers.aftersales_server'
        Write-Host 'MCP services are ready. Fill .env to start the main application.'
        exit 0
    }

    foreach ($name in @('CHAT_BASE_URL','CHAT_MODEL','CHAT_API_KEY','EMBED_API_KEY','RERANK_API_KEY')) {
        if (!$values[$name] -or $values[$name] -eq 'sk-xxx') {
            Write-Host "Please fill $name in .env, save, and run this shortcut again."
            throw 'Model configuration is incomplete. No API key has been printed.'
        }
    }
    Write-Host '[3/5] Preparing knowledge base...'
    Invoke-Checked $python @('-m','scripts.build_kb')
    Invoke-Checked $python @('-m','scripts.vectorize_kb')
    Write-Host '[4/5] Starting logistics and aftersales services...'
    Start-ServiceProcess 'mcp-logistics' 8101 '-m mcp_servers.logistics_server'
    Start-ServiceProcess 'mcp-aftersales' 8102 '-m mcp_servers.aftersales_server'
    Write-Host '[5/5] Starting CareMate...'
    Start-ServiceProcess 'mewhelp-app' 8000 '-m uvicorn app.main:app --host 127.0.0.1 --port 8000'
    if (Test-Path -LiteralPath (Join-Path $root 'data\ch10\onnx\model.onnx')) {
        Start-ServiceProcess 'topic-classifier' 8110 '-m scripts.ch10.serve'
    } else {
        Write-Host 'Classifier artifacts have not been prepared yet (ch10-train and ch10-export).'
    }
    $pages = @('http://localhost:8000/', 'http://localhost:8000/admin')
    foreach ($page in $pages) {
        $response = Invoke-WebRequest $page -UseBasicParsing -TimeoutSec 15
        if ($response.StatusCode -ne 200) { throw "CareMate page check failed: $page" }
    }
    Write-Host 'CareMate chat is ready: http://localhost:8000/'
    Write-Host 'CareMate admin is ready: http://localhost:8000/admin'
    if (!$NoBrowser) {
        foreach ($page in $pages) { Start-Process $page }
    }
} catch {
    foreach ($name in $script:startedNames) {
        $owned = Get-OwnedProcess $name
        if ($owned) { & taskkill.exe /PID $owned.Id /T /F | Out-Null }
        Remove-Item -LiteralPath (Join-Path $stateDir "$name.json") -ErrorAction SilentlyContinue
    }
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
