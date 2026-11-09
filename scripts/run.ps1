$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = (Join-Path $Root "src")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "请先运行 .\scripts\setup_env.ps1" }
& $Python -m pcb_ar @args
exit $LASTEXITCODE
