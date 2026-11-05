$QWEN3_P6B_MODEL = Join-Path $HOME 'Desktop\\PROGRAMS\\cuda\\models\\Qwen3-0.6B'
$QWEN35_35B_FP8_MODEL = Join-Path $HOME 'Desktop\\PROGRAMS\\cuda\\models\\Qwen3.5-35B-A3B-FP8'

# $MODEL = $QWEN3_P6B_MODEL
$MODEL = $QWEN35_35B_FP8_MODEL

$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot 'build\windows\Release\nano-vllm-cpp-bench.exe'

$timestamp = Get-Date -Format 'yyyyMMddHHmm'
$outputDir = Join-Path $projectRoot 'profile'
$outputFile = Join-Path $outputDir "qwen35_$timestamp.json"

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

& $executable `
    --model $MODEL `
    --input-tokens 128 `
    --output-tokens 128 `
    --batch-sizes 1 `
    --warmup 1 `
    --repeat 3 `
    --num-blocks 32 `
    --block-size 16 `
    --max-batched-tokens 128 `
    --cpu-threads 7 --cpu-affinity `
    --gpu-expert-ids 0,2,4,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66 `
    --expert-adapt-every 4 `
    --output $outputFile
exit $LASTEXITCODE
