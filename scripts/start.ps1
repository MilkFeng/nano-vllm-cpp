$QWEN3_P6B_MODEL = Join-Path $HOME 'Desktop\\PROGRAMS\\cuda\\models\\Qwen3-0.6B'
$QWEN35_35B_FP8_MODEL = Join-Path $HOME 'Desktop\\PROGRAMS\\cuda\\models\\Qwen3.5-35B-A3B-FP8'

# $MODEL = $QWEN3_P6B_MODEL
$MODEL = $QWEN35_35B_FP8_MODEL

$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot 'build\windows\Release\nano-vllm-cpp-cli.exe'

& $executable `
    --model $MODEL `
    --prompt '你好' `
    --max-tokens 256 `
    --num-blocks 256
exit $LASTEXITCODE
