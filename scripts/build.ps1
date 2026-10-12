$ErrorActionPreference = 'Stop'

if (-not $env:VCPKG_ROOT) {
    $env:VCPKG_ROOT = Join-Path $HOME 'vcpkg'
    Write-Host "VCPKG_ROOT not set, using default location: $env:VCPKG_ROOT"
} else {
    Write-Host "Using VCPKG_ROOT: $env:VCPKG_ROOT"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDirectory = Join-Path $projectRoot 'build\windows\Release'
$toolchain = Join-Path $env:VCPKG_ROOT 'scripts\buildsystems\vcpkg.cmake'

cmake -S $projectRoot -B $buildDirectory -G Ninja `
    '-DCMAKE_BUILD_TYPE=Release' `
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
    '-DVCPKG_TARGET_TRIPLET=x64-windows' `
    '-DCMAKE_CUDA_ARCHITECTURES=native'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $buildDirectory --config Release --parallel
exit $LASTEXITCODE
