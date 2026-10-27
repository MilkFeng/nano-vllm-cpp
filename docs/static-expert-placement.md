# 静态专家放置与 benchmark

当前实现固定每层的 GPU 专家名单，未驻留的专家在 CPU 执行；共享专家始终在
GPU。配置加载后不换入换出，不修改 Router 的 top-k 结果或路由权重。
配置生成与扫参脚本使用 Python 3.9+，无需安装额外的 Python 包。

## 1. 收集路由统计

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
  传输、合并与共享专家，未拆出各设备的独立计算时间。
- `experts`：被调用专家的 ID、累计 `tokens`、`calls` 和 `group_sizes`。
  例如 `"group_sizes": {"1": 5, "4": 2}` 表示该专家收到过 5 次单行和 2 次四行输入，
  合计 13 个 token；这里记录 executor 聚合后的完整 `M_e`，不是 CPU 的 4/2/1 微块。

每次 generation 单独重置统计；预热不写入报告。开启 profiling 会产生统计开销，
请把它用于校准，正式性能比较关闭 `--profile-experts`。

## 2. 生成静态配置

每层选择最常用的 32 个专家：

```bash
python3 scripts/expert_placement.py plan --profile calibration.json \
  --experts-per-layer 32 --phase mixed --output placement.json
```

在全模型 4096 MiB 的 routed 专家缓存预算内，按 decode 路由频率全局选择
`(层, 专家)`，不同层可以得到不同数量的槽位：

```bash
python3 scripts/expert_placement.py plan --profile calibration.json \
  --budget-mib 4096 --phase decode --output placement-decode.json
```

`--phase prefill|decode|mixed` 控制选择依据。mixed 先将两个阶段的分配频率分别
归一化，再用 `--prefill-weight` 加权，默认各占 0.5。权重表示两个阶段的偏好，
不是预估的延迟收益。某个所需阶段没有样本时会报错，例如输出只有一个 token
时不能生成 decode 方案。`--profile a.json b.json` 可合并维度相同的多个校准报告；
使用者应确保它们来自同一 checkpoint。频率与分配覆盖率无法替代实际耗时测量。

所有 routed 专家尺寸相同，预算按实际 FP8 权重和 FP32 scale 字节数向下取整；
超出全部专家容量的预算会截断。专家 ID 相同分数时按层号、ID 升序，配置可复现。
`--strategy uniform` 选择每层从 0 开始的 ID，作为相同槽位数的基线。
全局预算不能整除层数时，uniform 将剩余槽位依次分给前面的层。

**预算只包含 routed 专家缓存**，不包含其余模型权重、KV Cache、循环状态及工作区。
35B 的每个专家约 3 MiB，40 层每层 32 个约 3.75 GiB。可运行的预算应根据目标
输入长度和 batch 的实际显存占用确定。

## 3. 比较相同预算的放置方案

```bash
python3 scripts/expert_placement.py sweep --profile calibration.json \
  --counts 3,16,32 --phase mixed \
  --bench-exe ./build/nano-vllm-cpp-bench --output-dir placement-bench \
  -- --model /path/to/Qwen3.5-35B-A3B-FP8 \
  --input-tokens 128 --output-tokens 128 --batch-sizes 1 \
  --num-blocks 32 --block-size 16 --max-batched-tokens 128 \
  --cpu-threads 8 --warmup 1 --repeat 4 \
  --prompts-file examples/expert-calibration-prompts.json
```

这会生成并运行 6 组配置：每层 3/16/32 个专家，各自比较 uniform 与 hot。
将 `--counts` 换成 `--budgets-mib 1024,2048,4096` 可比较全局缓存预算分配，
默认两个策略都运行，`--strategies hot` 可只运行热点方案。
每个候选重新加载模型，模型加载耗时独立记录，不包含在推理耗时中。
`--` 后传普通 benchmark 参数；输出路径、放置参数和 profiling 由脚本管理。

输出目录包含每个候选的 `.config.json`、`.bench.json` 和汇总的 `comparison.json`。
汇总记录 prefill 时间中位数、decode 吞吐均值、TTFT/TPOT 中位数，以及相同预算、
相同输入/输出/batch 下相对 uniform 的倍率。单个候选失败会记录退出码并继续，
整个 sweep 最终以非零状态退出。OOM 后可以从汇总中确定较低的可运行预算。

校准覆盖率来自原始路由统计，不是新方案的实际 GPU 命中率。
`memory_after_load` 只是加载模型及缓存后的显存快照，不是推理峰值。
评估放置方案泛化效果时，将 sweep 的 prompt 文件换为另一份代表性文本。

Windows PowerShell 下使用 `python scripts/expert_placement.py ...`，
`--bench-exe .\build\windows\Release\nano-vllm-cpp-bench.exe`，
模型路径和其他参数不变；多行命令的续行符改为反引号。

## 4. 手工配置与正常推理

```json
{
  "schema_version": 1,
  "num_hidden_layers": 40,
  "num_experts": 256,
  "default": [0, 2, 4],
  "layers": {
    "0": [5, 19, 87],
    "1": [2, 41, 128, 203],
    "2": []
  }
}
```

`layers` 用十进制层号覆盖 `default`；未提供 default 时默认空名单。
空名单表示该层所有 routed 专家在 CPU 上计算。示例 ID 只展示格式，没有经过校准。
文件中的层数和专家数可省略，提供时必须匹配模型。非法层号、专家越界、重复 ID
均在分配模型权重前拒绝。

CLI 和 benchmark 都支持：

```text
--gpu-expert-config placement.json
```

它与旧的 `--gpu-expert-ids` 互斥。旧参数仍表示每层使用相同名单。
C++ API 使用 `EngineParams::gpu_expert_ids_by_layer`，非空时必须包含每层的名单，
并保持 `gpu_expert_ids` 为空。也可直接设置 `EngineParams::gpu_expert_config` 为
配置文件路径，与两种显式 ID 名单互斥；文件解析和校验由库完成。
benchmark 的 `expert_placement.layers` 始终记录解析
后的完整名单，以及总槽位数、专家字节数与缓存字节数，便于复核。
