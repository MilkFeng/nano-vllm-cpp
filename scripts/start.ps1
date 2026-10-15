$QWEN3_P6B_MODEL = Join-Path $HOME 'Desktop\\PROGRAMS\\cuda\\models\\Qwen3-0.6B'
$QWEN35_35B_FP8_MODEL = Join-Path $HOME 'Desktop\\PROGRAMS\\cuda\\models\\Qwen3.5-35B-A3B-FP8'

# $MODEL = $QWEN3_P6B_MODEL
$MODEL = $QWEN35_35B_FP8_MODEL

$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot 'build\windows\Release\nano-vllm-cpp-cli.exe'

& $executable `
    --model $MODEL `
    --prompt 'Write a Python function is_palindrome_ignore_non_alnum(s: str) -> bool that checks if a string is a palindrome, ignoring all non-alphanumeric characters and case. Provide test cases including edge cases like empty strings and punctuation-only strings.' `
    --max-tokens 1024 `
    --num-blocks 256 `
    --max-seqs 1 `
    --gpu-expert-ids 0,2,4 `
    --seed 42 `
    --repeat 3
exit $LASTEXITCODE
