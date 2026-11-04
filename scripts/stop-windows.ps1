param([switch]$ClassifierOnly, [switch]$ToolsOnly)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root '.venv\Scripts\python.exe'
$stateDir = Join-Path $root 'data\windows-launcher'
$names = if ($ClassifierOnly) { @('topic-classifier') } elseif ($ToolsOnly) { @('mcp-logistics','mcp-aftersales') } else { @('mewhelp-app','mcp-logistics','mcp-aftersales','topic-classifier') }
foreach ($name in $names) {
    $file = Join-Path $stateDir "$name.json"
    if (!(Test-Path -LiteralPath $file)) { Write-Host "$name has no saved process record."; continue }
    $saved = Get-Content -LiteralPath $file -Raw | ConvertFrom-Json
    $process = Get-Process -Id $saved.Id -ErrorAction SilentlyContinue
    if ($process) {
        $processPath = $process.Path
        # GNU Make may invoke 32-bit PowerShell; CIM reads 64-bit executable paths too.
        if (!$processPath) {
            $processPath = (Get-CimInstance Win32_Process -Filter "ProcessId = $($saved.Id)" -ErrorAction Stop).ExecutablePath
        }
        if ($process.StartTime.ToUniversalTime().Ticks.ToString() -ne $saved.StartTicks -or $processPath -ne $python) {
            throw "Unable to verify $name (PID $($saved.Id), path '$processPath', expected '$python'). Its saved process record has been retained."
        }
        # Windows venv python.exe starts a child interpreter. Stop only this validated process tree.
        & taskkill.exe /PID $process.Id /T /F | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Unable to stop $name. Its saved process record has been retained." }
        Write-Host "$name stopped."
    }
    Remove-Item -LiteralPath $file
}
Write-Host 'CareMate application services stopped. Database containers and data are retained.'
