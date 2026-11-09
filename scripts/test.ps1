$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = (Join-Path $Root "src")
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = (Join-Path $env:WINDIR "Fonts")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "请先运行 .\scripts\setup_env.ps1" }
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m pcb_ar --self-test --report (Join-Path $Root "artifacts\selftest_report.json") --screenshot (Join-Path $Root "artifacts\demo_main.png")
exit $LASTEXITCODE
