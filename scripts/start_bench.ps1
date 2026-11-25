param(
    [string]$Model = (Join-Path $HOME 'Desktop\PROGRAMS\cuda\models\Qwen3.5-35B-A3B-FP8'),
    [string]$InputTokens = '512',
    [string]$OutputTokens = '512',
    [string]$BatchSizes = '1',
    [int]$Warmup = 2,
    [int]$Repeat = 5,
    [int]$NumBlocks = 128,
    [int]$BlockSize = 16,
    [int]$MaxBatchedTokens = 128,
    [int]$CpuThreads = 7,
    [switch]$NoCpuAffinity,
    [AllowEmptyString()]
    [string]$GpuExpertIds = '0,2,4,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100',
    [int]$ExpertAdaptEvery = 4,
    [string]$PromptsFile = '',
    [string]$OutputFile = '',
    [string]$Executable = ''
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
if (-not $Executable) {
    $Executable = Join-Path $projectRoot 'build\windows\Release\nano-vllm-cpp-bench.exe'
}
if (-not $PromptsFile) {
    $PromptsFile = Join-Path $projectRoot 'examples\benchmark-prompts.json'
}
if (-not $OutputFile) {
    $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $OutputFile = Join-Path $projectRoot "profile\benchmark_$timestamp.json"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputFile)) {
    $OutputFile = Join-Path $projectRoot $OutputFile
}

$outputDir = Split-Path -Parent $OutputFile
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$benchmarkArgs = @(
    '--model', $Model,
    '--input-tokens', $InputTokens,
    '--output-tokens', $OutputTokens,
    '--batch-sizes', $BatchSizes,
    '--warmup', $Warmup,
    '--repeat', $Repeat,
    '--num-blocks', $NumBlocks,
    '--block-size', $BlockSize,
    '--max-batched-tokens', $MaxBatchedTokens,
    '--cpu-threads', $CpuThreads,
    '--expert-adapt-every', $ExpertAdaptEvery,
    '--seed', 42,
    '--prompts-file', $PromptsFile,
    '--output', $OutputFile
)
if (-not $NoCpuAffinity) {
    $benchmarkArgs += '--cpu-affinity'
}
if ($GpuExpertIds) {
    $benchmarkArgs += @('--gpu-expert-ids', $GpuExpertIds)
}

& $Executable @benchmarkArgs
exit $LASTEXITCODE
