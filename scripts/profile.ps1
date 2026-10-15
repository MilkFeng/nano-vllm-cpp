$ErrorActionPreference = 'Stop'
$startScript = Join-Path $PSScriptRoot 'start.ps1'
$report = Join-Path $env:TEMP ("nano-profile-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))

$nsys = (Get-Command nsys -ErrorAction SilentlyContinue).Source
if (-not $nsys) {
    $nsys = Get-ChildItem "$env:ProgramFiles\NVIDIA Corporation\Nsight Systems*\target-windows-x64\nsys.exe" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}

& $nsys profile `
    --trace 'cuda,cublas,nvtx' `
    --sample=none `
    --cpuctxsw=none `
    -o $report `
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File $startScript
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $nsys stats `
    --report 'cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum' `
    "${report}.nsys-rep" |
    Tee-Object -FilePath "${report}-summary.txt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nReport: ${report}.nsys-rep"
Write-Host "Summary: ${report}-summary.txt"
