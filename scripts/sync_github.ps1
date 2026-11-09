# 一键同步 obsai2026 到 GitHub。
# 令牌只从环境变量 GITHUB_TOKEN 读取，不写进脚本、不写进 git remote URL。
[CmdletBinding()]
param(
    [string]$Owner = "biekuaicom",
    [string]$Repo = "obsai2026",
    [string]$Branch = "main",
    [switch]$Private
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Token = $env:GITHUB_TOKEN
if ([string]::IsNullOrWhiteSpace($Token)) {
    throw "缺少环境变量 GITHUB_TOKEN。`n请先在当前 PowerShell 设置：`n  `$env:GITHUB_TOKEN = '<your-token>'`n然后重新运行本脚本。"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "未找到 git，请先安装 Git for Windows。"
}

Write-Host "== obsai2026 GitHub 同步 ==" -ForegroundColor Cyan
Write-Host "本地目录: $Root"

# 1. 初始化 git（首次运行）
if (-not (Test-Path (Join-Path $Root ".git"))) {
    Write-Host "[1/5] 初始化本地仓库..."
    & git init -b $Branch | Out-Host
    if ($LASTEXITCODE -ne 0) {
        & git init | Out-Host
        & git checkout -b $Branch | Out-Host
    }
} else {
    Write-Host "[1/5] 本地仓库已存在"
}

# 2. 本地提交身份（只影响本仓库）
if ([string]::IsNullOrWhiteSpace((& git config --get user.name))) {
    & git config user.name $Owner
}
if ([string]::IsNullOrWhiteSpace((& git config --get user.email))) {
    & git config user.email "$Owner@users.noreply.github.com"
}

# 3. 安全门：绝不能把 .venv 提交进去
$null = & git check-ignore -q ".venv"
if ($LASTEXITCODE -ne 0) {
    throw "安全中止：.gitignore 没有忽略 .venv，请先修复 .gitignore。"
}
$trackedVenv = & git ls-files -- ".venv"
if ($trackedVenv) {
    throw "安全中止：检测到已跟踪的 .venv 文件，请先 git rm -r --cached .venv。"
}

# 4. 暂存、提交
Write-Host "[2/5] 暂存并提交..."
& git add -A
if ($LASTEXITCODE -ne 0) { throw "git add 失败。" }

& git diff --cached --quiet
$hasStaged = ($LASTEXITCODE -ne 0)
if ($hasStaged) {
    $message = "chore: sync obsai2026 $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    & git commit -m $message | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "git commit 失败。" }
} else {
    Write-Host "没有新的改动需要提交"
}

$currentBranch = (& git rev-parse --abbrev-ref HEAD).Trim()
if ($currentBranch -ne $Branch) {
    & git branch -M $Branch
}

# 5. 用 GitHub API 检查/创建仓库
Write-Host "[3/5] 检查 GitHub 仓库 $Owner/$Repo..."
$headers = @{
    Authorization = "Bearer $Token"
    Accept = "application/vnd.github+json"
    "User-Agent" = "obsai2026-sync"
}
$api = "https://api.github.com/repos/$Owner/$Repo"
$repoInfo = $null
try {
    $repoInfo = Invoke-RestMethod -Uri $api -Headers $headers -Method Get -TimeoutSec 45
    Write-Host "仓库已存在: $($repoInfo.html_url)"
} catch {
    $status = 0
    if ($_.Exception.Response) {
        $status = [int]$_.Exception.Response.StatusCode
    }
    if ($status -ne 404) {
        throw "GitHub API 查询失败（HTTP $status）：$($_.Exception.Message)"
    }
    Write-Host "仓库不存在，创建新仓库..."
    $payload = @{
        name = $Repo
        description = "PCB microscope live AR annotation and verification desktop app"
        private = [bool]$Private
        has_issues = $true
        has_wiki = $false
        auto_init = $false
    } | ConvertTo-Json -Depth 4
    $repoInfo = Invoke-RestMethod -Uri "https://api.github.com/user/repos" `
        -Headers $headers -Method Post -Body $payload -ContentType "application/json" -TimeoutSec 60
    Write-Host "已创建: $($repoInfo.html_url)"
}

# 6. 设置 remote（URL 不含 token），推送
$remoteUrl = "https://github.com/$Owner/$Repo.git"
$remotes = @(& git remote)
if ($remotes -contains "origin") {
    & git remote set-url origin $remoteUrl
} else {
    & git remote add origin $remoteUrl
}

Write-Host "[4/5] 推送 $Branch ..."
$basic = [Convert]::ToBase64String(
    [Text.Encoding]::ASCII.GetBytes("x-access-token:$Token")
)
$oldPrompt = $env:GIT_TERMINAL_PROMPT
$env:GIT_TERMINAL_PROMPT = "0"
try {
    & git -c "credential.helper=" -c "http.extraheader=AUTHORIZATION: Basic $basic" `
        push -u origin $Branch
    if ($LASTEXITCODE -ne 0) {
        throw "git push 失败。请确认 token 有 repo 权限，且远端没有需要先 pull 的提交。"
    }
} finally {
    $env:GIT_TERMINAL_PROMPT = $oldPrompt
}

Write-Host "[5/5] 完成" -ForegroundColor Green
Write-Host "GitHub: $($repoInfo.html_url)" -ForegroundColor Green
Write-Host "Clone : $remoteUrl"
