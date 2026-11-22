# 编译 release 并打出 Windows 发布包 target\dist\stv-<版本>-windows-x86_64.zip：
# stv.exe（窗口程序）、stv.com（命令行用的控制台版）、README.md、LICENSE，
# 以及 ffmpeg\（ffmpeg.exe、同目录的 DLL、它的许可和源码说明）。
#
# 用法：pwsh -File scripts/package.ps1 [-Ffmpeg <ffmpeg.exe 路径>]
# 默认使用 PATH 中的 ffmpeg。共享库构建的 DLL 与 ffmpeg.exe 在同一目录，会一起复制；
# gyan.dev 等构建把 LICENSE、README.txt 放在 bin 的上一级，找到就一起放进去。
param([string]$Ffmpeg = (Get-Command ffmpeg -CommandType Application -ErrorAction Stop).Source)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

cargo build --release
if ($LASTEXITCODE) { exit $LASTEXITCODE }

$version = (Select-String -Path Cargo.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$name = "stv-$version-windows-x86_64"
$stage = "target\dist\$name"
$zip = "$stage.zip"
foreach ($old in $stage, $zip) {
    if (Test-Path $old) { Remove-Item -Recurse -Force $old }
}
New-Item -ItemType Directory -Force "$stage\ffmpeg" | Out-Null
Copy-Item target\release\stv.exe, README.md, LICENSE $stage

# stv.com 与 stv.exe 只差 PE 头里的子系统（IMAGE_OPTIONAL_HEADER.Subsystem，2 = 窗口，3 = 控制台）。
# 终端里输入 stv 时按 PATHEXT 先找到 .com，终端会等它结束并显示输出；双击的 stv.exe 不弹终端。
$dir = (Resolve-Path $stage).Path   # [IO.File] 不认 Set-Location，用绝对路径
$bytes = [IO.File]::ReadAllBytes("$dir\stv.exe")
$subsystem = [BitConverter]::ToInt32($bytes, 0x3C) + 0x5C
if ($bytes[$subsystem] -ne 2) { throw "stv.exe 不是窗口程序（子系统 = $($bytes[$subsystem])）" }
$bytes[$subsystem] = 3
[IO.File]::WriteAllBytes("$dir\stv.com", $bytes)

$bin = Split-Path -Parent (Resolve-Path $Ffmpeg)
Copy-Item $Ffmpeg "$stage\ffmpeg"
Get-ChildItem $bin -Filter *.dll | Copy-Item -Destination "$stage\ffmpeg"
$notices = foreach ($dir in $bin, (Split-Path -Parent $bin)) {
    Get-ChildItem $dir -File | Where-Object Name -Match '^(LICENSE|COPYING|README)'
}
if (-not $notices) { throw "在 $bin 及其上一级没找到 ffmpeg 的许可文件，FFmpeg 的许可要求随二进制一起分发" }
$notices | Copy-Item -Destination "$stage\ffmpeg"

& "$stage\ffmpeg\ffmpeg.exe" -hide_banner -version | Select-Object -First 1
Compress-Archive -Path $stage -DestinationPath $zip
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
"{0}  {1}  ({2:N1} MB)" -f $hash, (Split-Path -Leaf $zip), ((Get-Item $zip).Length / 1MB)
