# 固定显存预算下的在线专家替换

在线替换在现有 GPU routed 专家槽位中更新专家身份。默认关闭，
`--gpu-expert-ids` 指定各层相同的初始名单，名单长度决定每层槽位数。
例如 `--gpu-expert-ids 0,2,4` 为每层分配 3 个槽位，之后由在线热度更新其中的专家。
推理期间不增加或跨层转移槽位，不重新分配 GPU 专家权重；预算包含 FP8 权重
及 FP32 scales。其他模型工作区和 KV cache 不在这个预算内。

CLI 与 benchmark 都支持：

```text
--expert-adapt-every 4 --expert-adapt-swaps 8 --expert-adapt-decay 0.7
```

- `--expert-adapt-every`：每 N 个 decode 调度步调整一次，默认 0（静态）。
  一个多请求 decode batch 也只算一步；prefill 与分块 prefill 均不计入。
- `--expert-adapt-swaps`：每轮全模型最多替换几个专家，默认 8，必须大于 0。
- `--expert-adapt-decay`：每轮调整后热度乘数，默认 0.7，范围 [0, 1]。

例如：

```bash
./build/nano-vllm-cpp-bench --model /path/to/Qwen3.5-35B-A3B-FP8 \
  --gpu-expert-ids 0,2,4 \
  --expert-adapt-every 4 --expert-adapt-swaps 8 --expert-adapt-decay 0.7 \
  --input-tokens 128 --output-tokens 128 --batch-sizes 1 \
  --num-blocks 32 --block-size 16 --max-batched-tokens 128 \
  --cpu-threads 8 --warmup 1 --repeat 3 --output adaptive.json
```

每个 decode token–expert 分配为该层该专家的热度加 1，不使用路由权重作分数。
每层将非驻留专家按热度降序排列，驻留专家按热度升序排列，一一配对。
候选热度至少 2，且比被替换专家至少高 1.5，才形成一个替换。
随后按热度差在全模型排序，截取最多 `max_swaps` 对；相同分数按层号、
换入专家 ID、槽位号确定顺序。调整轮次即使没有发生替换也会衰减热度。
每层 0 个或全部专家驻留时，该层不会产生替换。

这一版只统计 decode 路由，prefill 仍使用当前驻留名单。旧专家的完整权重
保留在 RAM，因此淘汰 GPU 副本不需要 D2H。替换发生在本步全部 MoE 完成后，
使用 inference stream 同步上传每个专家的全部权重和 scales，再更新双向映射，
从下一步开始生效。复制失败会使受影响层的缓存失效并中止本次推理，避免
执行被部分覆盖的专家。Router 的 top-k ID、权重和合并位置不随策略修改；
两端浮点计算的舍入差异仍可能影响生成结果。

C++ API 对应 `EngineParams::expert_adaptation`，例如：

```cpp
params.expert_adaptation.every = 4;
params.expert_adaptation.max_swaps = 8;
params.expert_adaptation.decay = 0.7;
// C++ 还可调整候选最低热度和最小热度差。
params.expert_adaptation.min_usage = 2.0;
params.expert_adaptation.min_gain = 1.5;
```

在线热度、调整步数和驻留名单属于模型状态，跨 `generate()` 保留，
与 `profile_experts` 无关。调用方需要重建 engine 才会恢复初始名单。
每次 `generate()` 只重置报告计数，不重置学习状态。

开启策略后 `GenerationMetrics::expert_adaptation` 以及 benchmark 每个 run 的
`expert_adaptation` 对象包含：

- `rounds`、`swaps`、`bytes_uploaded`：本次 generation 的调整轮数、
  实际替换数、上传字节数。
- `seconds`（JSON 为 `ms`）：选择及同步替换耗时，包含在 decode 总时间内，
  不包含在逐层 `expert_profile[].decode.moe_ms` 中。
- `gpu_assignments`、`cpu_assignments`：本次 generation 的 decode 分配数，
  不开启完整 profiling 也会记录；JSON 另有 `gpu_assignment_ratio`。
- `resident_ids_by_layer`：本次 generation 结束时各层的驻留名单。

CLI 加 `--metrics` 会打印替换数、传输量及耗时。benchmark 的
`settings.expert_adaptation` 记录策略参数；顶层 `expert_placement.layers`
记录初始名单，每个 run 的名单记录实际学习结果。

比较时保持初始配置、输入、输出、batch 和 CPU 线程数相同，分别使用
`--expert-adapt-every 0` 和非零值运行两个新 engine。预热及先前 case 的热度
会保留，预热轮数和 case 顺序也应相同。正式计时无需 `--profile-experts`。
命中率提高不保证延迟降低，应同时看 TPOT、TTFT 和替换耗时；增加
`max_swaps` 或减小 `every` 都会增加权重传输的机会。

构建及验证：

```bash
cmake -S . -B build -DNANOVLLM_BUILD_TESTS=ON
cmake --build build -j 8
ctest --test-dir build -R expert-adaptation --output-on-failure
```

策略测试无需 GPU；CUDA 测试覆盖槽位复用、FP8 权重与 scales、映射更新、
CPU/GPU 计算一致性、单行与多行输入，以及模型级 prefill/decode 和跨请求状态。
无可用 GPU 时 CUDA 测试标记为跳过。
