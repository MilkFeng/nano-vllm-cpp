# 专家路由统计

benchmark 的 `--profile-experts` 收集每层 prefill/decode 路由分布。
GPU 缓存通过 `--gpu-expert-ids` 初始化，运行时可以开启
[在线专家替换](online-expert-placement.md)。

```bash
./build/nano-vllm-cpp-bench --model /path/to/Qwen3.5-35B-A3B-FP8 \
  --input-tokens 128 --output-tokens 128 --batch-sizes 1 \
  --num-blocks 32 --block-size 16 --max-batched-tokens 128 \
  --cpu-threads 8 --gpu-expert-ids 0,2,4 --warmup 1 --repeat 4 \
  --prompts-file examples/expert-calibration-prompts.json \
  --profile-experts --output calibration.json
```

示例 prompt 文件用于验证流程，实际校准时请替换为代表自己负载的文本。
`--prompts-file` 接收 UTF-8 JSON 字符串数组，与 `--prompt` 互斥。
第 `r` 个正式轮次、第 `b` 个请求使用 `(r * batch_size + b) % 文本数` 的文本，
编码后截断或循环填充到指定输入长度。正式轮次从第 0 条文本重新开始，
因此不同预热次数不会改变正式轮次的输入。固定输入长度仍是合成负载。

报告中每个 `cases[].runs[].expert_profile[]` 对应一层，其 `prefill`、`decode`
分别包含：

- `calls`：该层实际执行的调度步数。
- `input_tokens`：实际处理的 token 数，包含分块 prefill 与重算。
- `gpu_assignments` / `cpu_assignments`：两端处理的 token–专家分配数。
- `moe_ms`：MoE 从入口到 CUDA stream 同步完成的 wall time，包含路由、专家、
  传输、合并与共享专家，未拆出各设备的独立计算时间。在线替换耗时单独记录在
  `expert_adaptation.ms` 中。
- `experts`：被调用专家的 ID、累计 `tokens`、`calls` 和 `group_sizes`。
  例如 `"group_sizes": {"1": 5, "4": 2}` 表示该专家收到过 5 次单行和 2 次四行输入，
  合计 13 个 token；这里记录 executor 聚合后的完整 `M_e`，不是 CPU 的 4/2/1 微块。

每次 generation 单独重置统计；预热不写入报告。开启 profiling 会产生统计开销，
正式性能比较关闭 `--profile-experts`。在线策略不依赖 profiling。

**专家预算只包含 routed 缓存**，不包含其他模型权重、KV cache、递推状态及工作区。
35B 的每个专家约 3 MiB；每层 3 个、40 层共约 360 MiB。
benchmark 的 `expert_placement` 记录初始完整名单、总槽位数、专家字节数和缓存
字节数；启用在线策略后，每个 run 的 `expert_adaptation.resident_ids_by_layer`
记录结束时的名单。
