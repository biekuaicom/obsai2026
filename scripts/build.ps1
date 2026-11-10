param(
    [switch]$SkipTests,
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "请先运行 .\scripts\setup_env.ps1" }
$env:PYTHONPATH = (Join-Path $Root "src")
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = (Join-Path $env:WINDIR "Fonts")

if (-not $SkipTests) {
    Write-Host "=== 1/4 运行测试 ==="
    & $Python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "测试失败，停止打包" }
}

Write-Host "=== 2/4 生成图标/清理旧产物 ==="
if (-not (Test-Path (Join-Path $Root "packaging\app.ico"))) {
    & $Python (Join-Path $Root "scripts\make_icon.py")
}
$Dist = Join-Path $Root "dist\PCB_AR_Annotator"
$Work = Join-Path $Root "build\pyinstaller"
if (Test-Path $Dist) { Remove-Item $Dist -Recurse -Force }
if (Test-Path $Work) { Remove-Item $Work -Recurse -Force }

Write-Host "=== 3/4 PyInstaller 打包（onedir） ==="
& $Python -m PyInstaller --noconfirm --clean --distpath (Join-Path $Root "dist") --workpath $Work (Join-Path $Root "packaging\pcb_ar.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 失败" }
$Exe = Join-Path $Dist "PCB_AR_Annotator.exe"
if (-not (Test-Path $Exe)) { throw "未找到打包后的 exe: $Exe" }

Write-Host "=== 4/4 打包后自检 ==="
$Artifacts = Join-Path $Root "artifacts"
New-Item -ItemType Directory -Force $Artifacts | Out-Null
$Report = Join-Path $Artifacts "packaged_selftest.json"
$Shot = Join-Path $Artifacts "packaged_demo.png"
Remove-Item $Report -ErrorAction SilentlyContinue
Remove-Item $Shot -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $Exe -ArgumentList @("--self-test", "--report", "`"$Report`"", "--screenshot", "`"$Shot`"") -PassThru -Wait
if ($proc.ExitCode -ne 0) {
    throw "打包后自检失败，exit=$($proc.ExitCode)，报告: $Report"
}
if (-not (Test-Path $Report)) { throw "打包后自检未生成报告: $Report" }
Write-Host "打包自检报告: $Report"

Copy-Item (Join-Path $Root "README.md") (Join-Path $Dist "README.md") -Force
Copy-Item (Join-Path $Root "THIRD_PARTY_NOTICES.md") (Join-Path $Dist "THIRD_PARTY_NOTICES.md") -Force
Copy-Item (Join-Path $Root "LICENSE") (Join-Path $Dist "LICENSE") -Force
if (Test-Path (Join-Path $Root "docs")) {
    Copy-Item (Join-Path $Root "docs") (Join-Path $Dist "docs") -Recurse -Force
}
$Quick = @"
PCB AR 显微标注核对系统 v0.2.3
================================
双击 PCB_AR_Annotator.exe 启动。

首次启动会自动生成演示工程并播放模拟显微镜画面。
若 Windows SmartScreen 提示，请选择“更多信息 -> 仍要运行”。

使用步骤:
1. 演示/摄像头/视频文件 启动实时画面。
2. 默认轻量扫描建图: 稳定拍照 + 红色覆盖 + 重叠校验。
   需要时再用 全图扫描（备用/增强）生成整图。
3. 自动检测候选 + 框选补漏 + 双击元件气泡快速改 3 个参数。
4. 右侧栏核对全部参数，标记已核对/异常。
5. Ctrl+S 保存工程，文件菜单导出 BOM。

工程/配置/日志位置:
%LOCALAPPDATA%\PCB_AR_Annotator

详细说明见 docs\03_使用指南.md，问题排查见 docs\05_打包发布.md。
"@
Set-Content -Path (Join-Path $Dist "使用说明.txt") -Value $Quick -Encoding UTF8

if (-not $SkipZip) {
    $Zip = Join-Path $Root ("dist\PCB_AR_Annotator_v0.2.3_win64.zip")
    if (Test-Path $Zip) { Remove-Item $Zip -Force }
    & $Python (Join-Path $Root "scripts\package_zip.py") $Dist $Zip
    if ($LASTEXITCODE -ne 0) { throw "压缩发布包失败" }
    Write-Host "压缩包: $Zip"
    $hash = Get-FileHash $Exe -Algorithm SHA256
    $zipHash = Get-FileHash $Zip -Algorithm SHA256
    @(
        "PCB AR Annotator v0.2.3",
        "exe: $($hash.Hash)",
        "zip: $($zipHash.Hash)"
    ) | Set-Content -Path (Join-Path $Root "dist\CHECKSUMS.txt") -Encoding UTF8
}

Write-Host ""
Write-Host "完成: $Dist"
