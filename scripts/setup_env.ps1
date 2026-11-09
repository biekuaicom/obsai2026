param(
    [string]$Python = "",
    [string]$IndexUrl = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $Python) {
    $candidates = @(
        "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312\python.exe",
        "C:\Program Files\Python312\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { $Python = $candidate; break }
    }
    if (-not $Python) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $Python = $cmd.Source }
    }
}
if (-not $Python -or -not (Test-Path $Python)) {
    throw "未找到 Python 3.12。请用 -Python 指定 python.exe 路径。"
}

$Venv = Join-Path $Root ".venv"
Write-Host "项目根目录: $Root"
Write-Host "使用系统 Python 创建隔离环境: $Python"
Write-Host "虚拟环境位置: $Venv"
Write-Host "不会修改系统 Python 或全局 site-packages。"

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    & $Python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "创建虚拟环境失败" }
}

$VenvPython = Join-Path $Venv "Scripts\python.exe"
$env:PIP_DEFAULT_TIMEOUT = "90"
& $VenvPython -m pip install --retries 5 --index-url $IndexUrl -r (Join-Path $Root "requirements-dev.txt")
if ($LASTEXITCODE -ne 0) { throw "安装依赖失败" }
& $VenvPython -m pip install -e $Root --no-deps --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "安装项目失败" }

Write-Host ""
Write-Host "完成。运行:"
Write-Host "  .\scripts\run.ps1"
Write-Host "  .\scripts\test.ps1"
