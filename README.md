# nano-vllm-cpp

使用 C++ 和 CUDA 实现 nano-vllm 的单 GPU 推理链路，支持 dense Qwen3 与文本 Qwen3.5 MoE。

## 构建

需要 CMake 3.28+、支持 C++20 的主机编译器和标准库、CUDA Toolkit、Rust/Cargo，以及 `vcpkg.json` 中列出的 C++ 依赖。首次配置会获取 tokenizers-cpp（含子模块）和 minja，构建 tokenizer 时还会下载 Rust crate。

### Windows 原生构建

安装 Visual Studio 2022 的“使用 C++ 的桌面开发”工作负载、Windows SDK 和“适用于 Windows 的 C++ CMake 工具”，以及 CUDA Toolkit、Git、Rust 的 `x86_64-pc-windows-msvc` 工具链。打开 **x64 Native Tools Command Prompt for VS 2022**，输入 `powershell` 进入继承该环境的 PowerShell，确保 `cmake`、`ninja`、`nvcc`、`git`、`cargo` 在 PATH 中。

在项目目录运行：

```powershell
$env:VCPKG_ROOT = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\vcpkg'
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

脚本只执行 CMake 配置和并行构建，默认 Release，使用 Ninja，因此不需要 CUDA 的 Visual Studio 构建集成。`VCPKG_ROOT` 未设置时默认使用 `$HOME\vcpkg`；`vcpkg.json` 已指定 baseline，CMake 会自动安装清单中的依赖。首次构建需能访问 GitHub 和 crates.io。

构建目录为 `build/windows/Release`，避免与原有 VS 生成器缓存冲突；可执行文件为 `build/windows/Release/nano-vllm-cpp-cli.exe`。CUDA 架构使用 `native` 自动检测本机 GPU（RTX 5060 Ti 为 `120`），也可修改脚本中的该参数。Windows 使用 `x64-windows` triplet 和动态 MSVC CRT，权重文件读取已适配 Windows 文件映射 API。

### Linux 构建

```bash
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

将架构 `120` 改为实际 GPU 对应值。C++ 和 CUDA 源文件均使用 C++20。

## 生成文本

模型目录需要包含 `config.json`、`tokenizer.json`、`tokenizer_config.json`，以及
`model.safetensors` 或 `model.safetensors.index.json` 与它列出的权重分片。程序按
配置中的 `architectures` 自动选择 Qwen3 或 Qwen3.5 MoE，无需手动指定模型类型。

```bash
./build/nano-vllm-cpp-cli --model "$HOME/models/Qwen3-0.6B" \
  --prompt 'Hello' \
  --temperature 0 --max-tokens 16 \
  --max-model-len 128 --max-seqs 4 --max-batched-tokens 128 \
  --num-blocks 16 --block-size 16
```

- `--prompt` 默认是一条用户消息，程序读取模型的 `chat_template` 并自动应用。重复 `--prompt` 可批量提交独立对话，输出顺序与输入一致。
- 默认开启思考模式；`--thinking=false` 关闭模型模板中的 `enable_thinking`。
- `--temperature 0` 使用 greedy 解码，正数使用温度采样；`--seed` 设置随机种子。
- `--max-batched-tokens` 限制每轮 token 数，较长输入会分块 prefill。
- `--num-blocks` 必填，指定全部请求共享的 KV 块池大小；`--block-size` 指定每块 token 数，默认 256。程序按指定数量分配，不再自动估算显存容量。
- `--cpu-threads 8` 设置 Qwen3.5 的 CPU 专家计算线程数，默认 8，必须为正数；所有层共用一个常驻线程池。
- `--gpu-expert-ids 0,2` 设置 Qwen3.5 各层初始常驻 GPU 的 routed 专家 ID，默认为空；其余专家在 CPU 计算。共享专家一直在 GPU，不计入名单。
- `--expert-adapt-every 4` 开启在线热点替换；`--expert-adapt-swaps 8` 限制每轮全模型替换数，`--expert-adapt-decay 0.7` 控制旧热度衰减。每层槽位数保持不变，默认关闭。
- `--json` 输出生成文本和 token ID；`--repeat 2` 在同一引擎中重复请求。Qwen3 保留前缀缓存，Qwen3.5 每次从零状态开始。
- 每次生成结束后，日志输出实际生成 token 数、耗时和 `tok/s`。耗时仅计入 `generate()`，包含输入编码、prefill、decode 和输出解码，不包含模型加载及聊天模板渲染；多条请求统计总输出吞吐，`--repeat` 的每轮单独记录。
- 达到 EOS、`--max-tokens` 或总长度 `--max-model-len` 时结束；`--ignore-eos` 可忽略 EOS。

PowerShell 中可同时显示并保存启动日志（含 `tok/s`）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 2>&1 | Tee-Object -FilePath run.log
```

KV 块数和 block size 必须大于零，块池至少容纳一条 `max_model_len` 长度的序列，总 token 容量不得超过 Int32 slot 索引范围。启动日志显示块数和全部 K/V 缓存的字节数。当前 dense Qwen3 的缓存字节数为 `2 * num_hidden_layers * num_blocks * block_size * num_key_value_heads * head_dim * sizeof(dtype)`；配置时还需给权重、激活和计算工作区留出显存，缓存分配成功不保证最大 batch 能运行。`--gpu-memory-utilization` 已移除。

C++ 调用方也必须设置 `EngineParams::num_kvcache_blocks`；默认值 `0` 会报参数错误。例如 `EngineParams params; params.num_kvcache_blocks = 16;`，配合默认 block size 256，可容纳一条默认长度 4096 的序列。

默认温度为 1，只做全词表温度采样；当前不会加载 `generation_config.json` 中的 temperature、top-k 或 top-p 配置。`--max-model-len` 限制总上下文长度；输出长度仍由 `--max-tokens` 控制，默认最多生成 64 个 token。

### Prefill / Decode benchmark

构建后生成独立的 `nano-vllm-cpp-bench`。它在同一个引擎上运行指定的输入长度、
输出长度和 batch 大小的笛卡尔积，每个配置先预热，再统计正式轮次。

```bash
./build/nano-vllm-cpp-bench --model "$HOME/models/Qwen3-0.6B" \
  --input-tokens 128,1024,4096 --output-tokens 128 --batch-sizes 1,4,8 \
  --warmup 1 --repeat 3 --num-blocks 2304 --block-size 16 \
  --max-batched-tokens 1024 --output benchmark.json
```

`--num-blocks` 是共享 KV 块池大小；上面的配置可容纳 8 条最大总长度为 4224
的序列。减小块池可能导致等待或抢占重算，实际执行的 prefill token 数会体现在
报告中。引擎的 `max_model_len` 自动设为最大输入长度加最大输出长度，
`max_seqs` 设为最大 batch 大小，因此整个矩阵使用相同的资源配置。

Qwen3.5 可以先用小输入、单序列测量，之后再增加长度和并发：

```bash
./build/nano-vllm-cpp-bench --model "$HOME/models/Qwen3.5-35B-A3B-FP8" \
  --input-tokens 128 --output-tokens 128 --batch-sizes 1 \
  --warmup 1 --repeat 3 --num-blocks 32 --block-size 16 \
  --max-batched-tokens 128 --cpu-threads 8 --gpu-expert-ids 0,2,4 \
  --output benchmark-qwen35.json
```

Windows 下对应的可执行文件为
`build\windows\Release\nano-vllm-cpp-bench.exe`，参数相同。例如 PowerShell：

```powershell
.\build\windows\Release\nano-vllm-cpp-bench.exe `
  --model "$HOME\Desktop\PROGRAMS\cuda\models\Qwen3.5-35B-A3B-FP8" `
  --input-tokens 128 --output-tokens 128 --batch-sizes 1 `
  --warmup 1 --repeat 3 --num-blocks 32 --block-size 16 `
  --max-batched-tokens 128 --cpu-threads 8 --gpu-expert-ids 0,2,4 `
  --output benchmark-qwen35.json
```

Qwen3.5 使用 `--gpu-expert-ids` 指定各层初始名单和缓存容量。
`--profile-experts` 在报告中记录
prefill/decode 各自的路由频率、专家聚合 token 数分布和整段 MoE 时间，预热轮次
不计入报告。该模式有统计开销，性能比较时应关闭。

统计格式及预算口径见[专家路由统计](docs/expert-profiling.md)。
固定缓存预算下的在线换入换出见[在线专家替换](docs/online-expert-placement.md)，
使用 `--expert-adapt-every N` 开启，默认关闭。

推理使用每个引擎独立的 CUDA 临时 Tensor 池，在相同容量预热后复用存储；
首次遇到更大容量时仍可能扩容，保留块计入显存预算，直到引擎销毁时释放。
单 token 的 CPU 专家结果按层打包上传。每层 3/8/16 个 GPU 专家槽位的前后对照、
显存峰值和正确性检查见[运行时性能记录](docs/runtime-workspace-performance.md)。
单 token MoE 合并路由与输入下载、共享专家提前提交和异步返回的实现与验证见
[MoE 同步优化](docs/moe-synchronization.md)。

计时口径：

| 指标 | 定义 |
| --- | --- |
| Prefill ms | 所有 prefill 调度步的 wall time 之和，包含分块 prefill 和抢占后的重算 |
| Prefill tok/s | 实际执行的 prefill 输入 token 数 / prefill 秒数；不把生成的首 token 计入分子 |
| Decode ms | 所有 decode 调度步的 wall time 之和 |
| Decode tok/s | 在 decode 步实际生成的 token 数 / decode 秒数；多序列时是聚合吞吐 |
| Decode ms/step | decode 总时间 / decode 步数；一个调度步可以为多条序列各生成一个 token |
| TTFT | 从整批请求提交到该请求第一个 token 在 CPU 上可用，包含排队和分块 prefill 等待 |
| TPOT | 每条请求的 `(最后 token 时间 - 首 token 时间) / (输出 token 数 - 1)`，包含期间的调度等待 |
| Output tok/s | 所有输出 token 数 / 整批 inference 时间，包含 prefill、decode 和调度开销 |

每个阶段的计时覆盖 `schedule → ModelRunner::run → postprocess`。
`run()` 返回前已经同步 CUDA stream，因此时间包含 GPU 实际执行、数据搬运与
CPU 专家计算等待。它是端到端执行阶段的 wall time，并非单纯 GPU kernel 时间。
第一个输出 token 通常由最后一个 prefill 块产生；中间 prefill 块丢弃的采样结果
不计入输出。请求在抢占后重新 prefill 时产生的新 token 也按实际阶段归类。
仅输出一个 token 时没有 decode，Decode tok/s 和 TPOT 在 JSON 中为 `null`，
终端显示 `n/a`。

模型加载、构造输入、聊天模板和输入分词不计入 inference。报告另存模型加载、
输出解码和整次调用时间。benchmark 固定 greedy、忽略 EOS，确保实际输出长度
与参数一致；预热和正式轮次都关闭前缀缓存，重复输入不会跳过 prefill。
输入由 `--prompt` 文本的 token ID 循环填充，逐请求旋转起始位置，得到精确长度，
不应用聊天模板。这是合成负载；尤其对 MoE，测试真实任务时应同时用 CLI 的
`--metrics` 观察实际提示词。

终端显示每个配置的阶段耗时与吞吐、TTFT p50/p95 和 TPOT p50。JSON 保存 GPU、
CUDA、构建类型、模型路径、全部配置、每轮原始数据、逐请求指标和均值、总体标准差、
min/p50/p95/max。阶段指标的样本是正式轮次，请求指标的样本是正式轮次中的全部
请求；百分位使用排序后的线性插值。每个配置完成后写入报告，全部完成后
`completed` 才置为 `true`。报告中的模型路径不代替权重版本标识，比较不同运行时
请保持权重和构建版本一致。

对真实文本使用现有 CLI：

```bash
./build/nano-vllm-cpp-cli --model "$HOME/models/Qwen3-0.6B" \
  --prompt 'Explain how paged attention works.' --thinking=false \
  --temperature 0 --max-tokens 128 --ignore-eos \
  --max-model-len 1024 --max-seqs 1 --max-batched-tokens 128 \
  --num-blocks 64 --block-size 16 --repeat 3 --metrics --no-prefix-cache
```

`--metrics` 只向 stderr 增加分阶段和逐请求日志，原有 `--json` 输出格式保持不变。
C++ 调用方可向 `generate()` 或 `generate_tokens()` 的第三个参数传入
`GenerationMetrics*` 获取原始秒数；不传入时不收集逐步指标。

可用小模型运行 GPU 集成检查，无需安装 Python 推理框架：

```bash
python3 tests/benchmark_test.py --model "$HOME/models/Qwen3-0.6B"
```

检查覆盖输入/输出/batch 矩阵、单 token 输出、分块 prefill、预热隔离、
重复运行不复用前缀、KV 不足时的抢占重算，以及原始计数、计时与汇总统计的关系。

### Qwen3.5 FP8 CLI

接入的目标是 `Qwen/Qwen3.5-35B-A3B-FP8`，使用下载后的原始分片，无需离线
转换。先用少量 KV 块、一个活跃序列和空 GPU 专家名单：

```bash
./build/nano-vllm-cpp-cli --model "$HOME/models/Qwen3.5-35B-A3B-FP8" \
  --prompt 'Hello' --thinking=false \
  --temperature 0 --max-tokens 8 \
  --max-model-len 128 --max-seqs 1 --max-batched-tokens 32 \
  --num-blocks 8 --block-size 16
```

`--max-seqs` 还决定 Qwen3.5 预分配的递推状态槽位数，并限制全部活跃序列数，
包含分块 prefill 中的序列。35B-A3B 当前每个槽位约 62.8 MiB；不要沿用 dense
Qwen3 的默认值 512，这会仅为 linear 状态分配约 31.4 GiB。日志分别显示 K/V
字节数、状态槽位与状态字节数、每层 GPU 专家数。可加 `--gpu-expert-ids 0,2`，
两个 FP32 routed 专家在 40 层共增加约 960 MiB 常驻权重。

这条命令给出运行接口，完整 35B 尚未在本机验证：当前 WSL 只有 22 GiB RAM，
仅全部 routed FP8 权重就约 30 GiB，需先让 WSL 使用足够的物理内存。模型加载
后非专家参数、激活和累加仍使用 FP32；GPU routed 专家保留 FP8 权重和 FP32
scale，计算时在 FP32 GEMM 内融合 FP8 权重反量化。
KV 和递推状态也在 GPU，尚未放到系统 RAM。

### C++ 聊天模板接口

[LLMEngine::apply_chat_template](include/nanovllm/nanovllm.hh) 使用 [minja](https://github.com/google/minja) 在 C++ 中渲染 checkpoint 的真实 Jinja 模板，运行时不依赖 Python。当前支持 system、user、assistant 的纯文本消息。

已有 `LLMEngine engine` 时，可以传入对话历史：

```cpp
const std::array messages{
    nanovllm::ChatMessage{"system", "You are a helpful assistant."},
    nanovllm::ChatMessage{"user", "Hello"},
};
const std::array prompts{
    engine.apply_chat_template(messages, {.enable_thinking = false}),
};
auto outputs = engine.generate(prompts, {.temperature = 0.0f});
```

模板接口默认 `add_generation_prompt=true`、`enable_thinking=true`，CLI 通过 `--thinking=false` 可关闭思考。库的 `generate()` 接受已格式化的 prompt，不会重复套模板。同一个引擎的 `generate()` 应串行调用；Qwen3 跨调用保留可复用的完整 KV 块，Qwen3.5 重新计算全部输入。

输出只包含新生成的 token。`<|im_start|>assistant` 已属于输入，不会回显；`<|im_end|>` 是新生成的正常结束标记，当前文本解码保留特殊 token。

## 代码组织

`src/layers/` 放通用的 `Linear`、`Embedding`、`RMSNorm`、`SwiGLUMLP`、`LMHead` 和 `RotaryEmbedding`。这些层接收具体维度，不依赖 `ModelConfig`；CUDA 算子和头文件放在 `src/kernels/cuda/`；CPU 算子放在 `src/kernels/cpu/`。

`src/models/qwen3.hh/.cc` 负责 Qwen3 Attention、DecoderLayer 和模型组装。`compute_logits()` 在 prefill 时选出每条序列最后一个 token 的 hidden state，再交给只做词表投影的 `LMHead`。`src/core/forward_context.hh` 保存由 ModelRunner 准备的执行元数据，包括 KV block table 和 CUDA stream。

`Qwen3Cache` 统一持有全部层的 K/V，两个 Tensor 的 shape 均为
`[num_hidden_layers, num_blocks, block_size, num_key_value_heads, head_dim]`，
dtype 与 Qwen3 模型一致。`key_cache(layer)` / `value_cache(layer)` 返回该层
的四维借用视图，`kv_bytes()` 统计 K/V 总字节数；它不注册为模型参数。
直接调用模型时，在加载权重后构造缓存并绑定：

```cpp
Qwen3Cache cache(config, /*num_blocks=*/128, /*block_size=*/16,
                 model.device());
model.transformer().set_cache(cache);
// Prepare ForwardContext metadata and execute prefill/decode.
// Before destroying the cache while retaining the model:
model.transformer().clear_cache();
```

缓存需覆盖所有使用它的 forward 的生命周期，释放前需等待相关 CUDA 操作完成。
新序列的 KV 可见范围由 block table 和 context length 控制，无需清零整个缓存。

## 已实现链路

```text
Token IDs → Embedding
  → [RMSNorm → Q/K/V Linear → Q/K Norm → RoPE
     → KV Cache → Causal Attention → O Linear
     → Residual + RMSNorm → Gate/Up Linear → SiLU × Up → Down Linear] × layers
  → Residual + RMSNorm → Last-token Gather → LM Head → Sampling → Token IDs
```

基础算子支持 FP32、FP16 和 BF16。Linear 使用 cuBLAS；attention 使用 FP32 在线 softmax，支持 GQA、变长 prefill、缓存前缀和分页 decode。引擎包含分块 prefill、KV 块管理、前缀缓存、显存不足时的请求抢占和停止条件处理。

内部 CUDA 算子只接受连续 Tensor，并使用调用线程已经选好的 device 0。通用模型配置约束，以及模型和算子内部的 shape、dtype 和索引约束使用 Debug 断言；Qwen3 支持范围、用户参数和 CUDA/cuBLAS 运行错误在 Release 中也会检查。缓冲区重叠关系由调用者保证。RoPE 直接原地修改 Q/K。

当前支持 CUDA device 0 上的 dense Qwen3 与文本 Qwen3.5 MoE，支持单文件及分片 safetensors、未缩放的 RoPE，以及偶数且不超过 256 的 full attention head dimension。Attention 是用于打通计算链路的基础实现；FlashAttention 级优化、CUDA Graph 和张量并行仍待实现。KV 块数由命令行或 `EngineParams` 手动指定。

FP16/BF16 的舍入会受 GEMM 的 batch 大小影响；候选分数接近时，生成 token 可能随批量配置变化，也不保证与 Transformers 逐 token 相同。当前融合 residual 和 RMSNorm 的计算顺序沿用 nano-vllm。

## FP8 权重读取与 CPU 反量化

`Checkpoint` 接受单个 safetensors 文件、`model.safetensors.index.json`，或模型
目录。目录中优先读取索引，否则读取 `model.safetensors`。索引只保存张量到分片
文件的映射；访问张量时才 mmap 对应分片，并保留映射直到 reader 析构。
未访问的分片可以尚未下载；访问缺失文件时报错，文件到位后可以重试。

`tensor(name)` 返回 dtype、shape 和偏移，`raw_data(name)` 返回只读字节 span。
这些引用和 span 借用 reader 的生命周期，打开其他分片不会使它们失效；调用需
串行。读取器不复制权重，也不自动反量化。`Model::load_ckpt()` 复用该读取器，
支持上述三种路径，继续严格检查名称、shape 和 dtype 后才复制参数。
Engine/CLI 按 architecture 调用相应模型，加载器接受模型目录以自动选择单文件
或分片索引。Qwen3.5 同时分配 full 层的 paged KV 和 linear 层的独立状态槽位。

新增 `DataType::Float8E4M3FN`，每元素 1 字节，对应 safetensors 的 `F8_E4M3`。
CPU `kernels::cpu::dequantize_fp8()` 将二维权重还原到调用方提供的 FP32 工作区：

```text
output[r,c] = decode_e4m3fn(weight[r,c]) * weight_scale_inv[r/Br,c/Bc]
```

默认 `Br=Bc=128`，也可传入其他正数。scale 必须是 FP32，形状为
`[ceil(R/Br), ceil(C/Bc)]`；尾块允许不足一个完整 block。三个矩阵需在 CPU 上、
连续、非空，输入与输出存储不能重叠。解码保留正负零、次正规数和 NaN，最大
有限绝对值为 448；它不解释 E4M3FNUZ 或 E5M2。

对于 scale 已经是 FP32 的 checkpoint，读取一个专家的 gate 矩阵可用：

```cpp
Checkpoint checkpoint("/path/to/fp8-checkpoint-with-f32-scales");
const std::string prefix =
    "model.language_model.layers.0.mlp.experts.0.gate_proj";
const auto& info = checkpoint.tensor(prefix + ".weight");
const auto bytes = checkpoint.raw_data(prefix + ".weight");
ConstTensorView weight(bytes.data(), info.dtype, TensorLayout(info.shape),
                       Device::cpu());

const auto& scale_info = checkpoint.tensor(prefix + ".weight_scale_inv");
auto scales = Tensor::empty(scale_info.shape, scale_info.dtype);
const auto scale_bytes = checkpoint.raw_data(prefix + ".weight_scale_inv");
std::memcpy(scales.data(), scale_bytes.data(), scale_bytes.size());
auto output = Tensor::empty(info.shape, DataType::Float32);
kernels::cpu::dequantize_fp8(weight, scales.view().as_const(), output.view());
```

FP8 权重直接借用 mmap 字节；scale 复制到 Tensor，保证浮点访问使用对齐的
存储。官方 Qwen3.5 FP8 权重的 scale 是 BF16，可用下面的模型专用加载器
转换。反量化函数用于加载时展开非专家 FP8 矩阵，输出由调用方提供；
CPU routed 专家的计算直接在点积内部解码。
`ExpertStore` 支持保存 FP8 权重和 scale，CPU 未命中专家在点积内部解码，GPU
固定缓存直接复制原始 FP8 权重和 FP32 scale。Qwen3.5 模型加载器可直接读取官方 FP8
checkpoint；GPU routed 专家在 FP32 GEMV/GEMM 内融合 FP8 权重反量化，
激活和累加保持 FP32。

## 学习用 MoE

`src/layers/moe.hh/.cc` 实现一个独立的 routed MoE 层，支持 FP32 的 `[T,H]`
输入，`T > 0`。维度由 `MoEConfig` 指定，例如 `H=128, I=64, E=16, K=2`。
`T` 是本次输入的 token 总数；每个 token 的 FFN 独立计算，可以来自不同请求，
不需要 attention 的序列边界或 KV 元数据。

```text
x[T,H] → Router Linear → logits[T,E]
       → 每个 token 独立 softmax、top-k、选中概率重新归一化
       → 按专家聚合 token，CPU/GPU 执行批量 SwiGLUMLP
       → 每个 token 按自己的路由权重求和 → y[T,H]
```

`MoE::forward()` 返回最终输出和以下中间结果：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `hidden_states` | `[T,H]` | routed 与可选共享分支的最终输出 |
| `router_logits` | `[T,E]` | softmax 之前的 Router 输出 |
| `expert_ids` | `[T,K]` | 每个 token 按 Router 排名选出的专家编号 |
| `expert_weights` | `[T,K]` | 每行独立归一化，和为 1 |
| `expert_outputs` | `[T*K,H]` | 第 `t*K+rank` 行是 token `t` 的第 `rank` 个专家原始输出 |
| `shared_expert_output` | `[T,H]` | 未乘门控的共享专家输出；关闭分支时为空 |
| `shared_gate_logits` | `[T,1]` | sigmoid 之前的共享门控；关闭分支时为空 |

专家输出按 token、Router 排名排列；路由权重只在 combine 时应用一次。
不同 token 可以选中同一专家。`T=1` 时保留原有形状，包括 `[K,H]` 的专家输出。
权重可通过 `Module::named_parameters()` 初始化；测试展示了完整的运行方式。

全部 routed 专家权重保存在 RAM 中的 `ExpertStore`，模型参数名仍为
`experts.N.gate_proj.weight` 等。GPU 的 `ExpertCache` 只分配初始名单对应数量的
专家槽位，缓存副本不注册为额外模型参数。Router 在 GPU 上计算，命中的专家
在 GPU 槽位上执行，未命中的专家在 CPU 上执行；CPU 输出复制回 GPU 后统一
combine。Router 的整批专家编号一次下载到 CPU，构造稳定的专家分组；有 CPU
专家时整批输入只下载一次，按专家打包后交给线程池。

```text
批量 Router → expert_ids[T,K]
            → ExpertExecutor::forward_into(input[T,H], ids[T*K])
                ├── GPU 专家组 → gather → FP8Expert(input[M_e,H])
                └── CPU 专家组 → 打包 → expert_forward(input[M_e,H])
            → 一次上传 CPU 输出 → scatter 恢复 expert_outputs[T*K,H]
            → CUDA combine → y[T,H]
```

`order[q]` 保存分组行对应的原始 `token*K+rank`。GPU 专家组在前，CPU 专家组
在后，CPU 分组输出因此可以一次连续上传。`moe_gather` 按 `order[q]/K` 收集
GPU 输入，`moe_scatter` 将完整的分组结果还原到原始行号，无需原子操作。
每个活跃专家收到 `[M_e,H]`，GPU 单行走 GEMV，多行走已有的 GEMM；CPU
按权重行遍历多个输入 token。路由权重仍只在原来的 combine 中应用一次。

`T=1` 直接向 `[K,H]` 输出写入，跳过分组排列和 gather/scatter，保留 decode
快速路径。executor 不再为每个 token 创建输出 Tensor 或额外做 D2D 汇总复制。

构造时传入共享线程池，固定驻留名单通过第四个构造参数传入。例如缓存 4 个专家：

```cpp
ThreadPool cpu_pool(8);
MoE moe({128, 64, 16, 2}, cpu_pool, Device::cuda(0),
        std::vector<std::int32_t>{0, 2, 4, 6});
```

初始化 `named_parameters()` 返回的权重后，调用
`moe.populate_expert_cache(stream)`，再调用 `moe.forward(x, stream)`。
routed 专家参数在 CPU，Router 参数在所选设备上；复制参数时按目标设备选择
`memcpy` 或 CUDA 复制。RAM 权重修改后需要再次填充缓存。

省略驻留名单（`std::nullopt`）缓存全部 routed 专家；传入显式空向量
`std::vector<std::int32_t>{}` 分配零个 GPU 专家槽位，全部选中 routed 专家在 CPU 执行。
CPU 构造只创建参考权重。`MoEOutput` 的 `gpu_experts`、`cpu_experts` 记录
整批 `T*K` 次 routed 专家选择的执行数量，不包含共享分支，同一专家被不同
token 选中会分别计数；`expert_cache().num_slots()`
和 `size_bytes()` 只统计 routed 缓存槽位和权重，不包含 Router、共享专家及工作区。

当前支持多 token、FP32 激活与 FP8 专家、1..1024 个专家、`1 <= top_k <= E`。
populate/replace 同步传入的 stream，调用需串行；forward 的 GPU 工作异步提交，
读取结果或释放模块前需等待相应 stream，开启 profiling 时仍等待本层完成。
缓存槽位数固定，
开启在线策略后会按 decode 路由热度更新槽位中的专家；
文本生成支持 Qwen3 与 Qwen3.5 MoE，后者使用下面的 FP8 RAM store。

### FP8 专家在 RAM 和 GPU 中存储

`ExpertStore(H, I, E)` 和 GPU 专家缓存统一保存 FP8 权重与 FP32 scale，
不提供 FP32 专家存储模式：

```cpp
MoEConfig config{
    .hidden_size = 2048,
    .intermediate_size = 512,
    .num_experts = 256,
    .top_k = 8,
    .shared_expert_intermediate_size = 512,
};
ThreadPool cpu_pool(8);
MoE moe(config, cpu_pool, Device::cuda(0), std::vector<std::int32_t>{0, 2, 4, 6});
```

每个专家注册六个 CPU 参数：三个 `gate_proj/up_proj/down_proj.weight` 为
FP8，三个同级 `weight_scale_inv` 为 FP32，按 `128x128` 分块。通过参数接口
初始化，或把模块注册到 `Model` 后用 `load_ckpt()` 加载；GPU 副本与计算工作区
不注册为参数。Router、共享专家、门控及激活仍为 FP32。官方 Qwen3.5 FP8
checkpoint 的 scale 也可以是 BF16，需要用下面的模型专用加载器转换；通用
`Model::load_ckpt()` 的默认实现只接受 dtype 相同的参数，Qwen3.5 模型会覆盖它
以处理 BF16 和 FP8 转换。

```text
ExpertStore: 全部专家 FP8 权重 + scale（RAM）
  ├── populate: 原样上传常驻 ID 的 FP8 权重 + FP32 scale → GPU 槽位
  └── dispatch 未命中: CPU 点积内部解码 FP8 → FP32 SwiGLU → 上传输出
```

CPU 未命中专家调用 `FP8Expert::forward_into(input, output)`，由
`kernels::cpu::expert_forward()` 直接读取 FP8 权重和 FP32 scale。
当前构建使用 AVX2/FMA，点积每次解码 8 个 FP8 值到 SIMD 寄存器并做浮点乘加。
权重始终保持 FP8，不创建完整的 FP32 权重副本。
每个计算线程复用一个中间激活缓冲。SIMD 的归约顺序和 FMA 会带来少量浮点误差，
不保证与标量实现逐位一致。

Qwen3.5 模型持有一个线程池，所有 MoE 层的 `ExpertExecutor` 共用它。
多 token 时每个活跃 CPU 专家组是一个任务：输入下载并打包后先提交 CPU 任务，
再提交 GPU 专家组，等待 CPU 完成后上传其输出，最后执行 combine。
异常退出也会等待所有已提交任务，
避免工作线程继续访问已经释放或复用的缓冲。
点积逐行、逐 128 列块计算；每个块读取一次 scale，用共享的 256 项 FP32
查表解码 FP8 值，立即做乘加，不创建完整 FP32 权重矩阵。查表占 1 KiB，
输入、SiLU、累加与输出保持 FP32，CPU 路径不额外量化激活。

CPU 点积逐行计算 gate/up，同一权重行连续服务多个 token，直接写出 SwiGLU
中间激活。每个计算线程通过 `thread_local` 复用 `[M_e,I]` 的 FP32 缓冲，
占 `M_e*I*sizeof(float)` 字节；单 token、I=512 时仍为 2 KiB。

CPU 专家按 4、2、1 行拆分 token。`fp8_dot_batch_kernel()` 对同一输出通道的
2/4 个 token 共用已解码、乘 scale 的 FP32 权重向量，每个 token 保持独立的
双累加器；gate/up 分别调用，再直接写入 activated，down 同样采用批量点积。
M=1 和剩余的单行继续调用原点积，不创建完整 FP32 权重矩阵或 gate/up 矩阵。
128 列 scale 块完整覆盖输入时使用无尾部快路径；其他尺寸保留 8 列及标量尾部，
输入 row stride 以 float 元素计。归约长度和中间激活/输出形状均不变。
编译器对尾部的向量化可能造成少量浮点差异，批量路径按容差验证。

缓存初始化只上传驻留名单，直接复制三个 FP8 权重和三个 FP32 scale，不申请
CPU 权重展开缓冲区。全部上传完成后才发布缓存映射。改变 RAM 权重或 scale
后需显式再次 populate。

GPU FP8 专家调用 `kernels::cuda::linear_fp8()`：加载权重时将 E4M3FN 解码成
FP32，再乘 checkpoint 中对应的 `weight_scale_inv[n/128,k/128]`。
反量化融合在 GEMM 内，输入激活直接以 FP32 参与计算，无需计算 amax 或
输入 scale，也不会增加激活量化误差。乘法、累加与输出都保持 FP32。

多行输入参考寄存器分块 GEMM：每个 block 计算 `16x64` 输出，每个线程
计算 `2x4` 小块，沿 K 以 32 为步长预取并双缓冲。FP8 权重加载后解码、
缩放并转置到 shared memory 的 `[K,N]` tile，与 FP32 输入做 `fmaf`。
`M=1` 时每个 warp 沿 K 并行计算一个输出，直接在寄存器里反量化权重。

一次 Linear 调用只启动一个 kernel，不创建完整 FP32 权重矩阵或全局量化
激活缓冲，也无需 BLAS workspace。M、N、K 可以不是 tile 的整数倍；
尾部加载做边界判断，不申请 padding 缓冲。GPU 缓存也直接保存原始 `[N,K]`
权重，无需补齐到 128 的倍数。executor 只保留 gate/up/activated 三个
FP32 中间张量。
CPU 与 GPU 都使用原始 FP32 激活，结果仍可能因累加顺序不同而有浮点差异。
未测量整模型加速。

令 `Sr=ceil(I/128), Sc=ceil(H/128)`，RAM 权重字节数为
`E*(3*H*I + 3*Sr*Sc*sizeof(float))`，GPU routed 权重和 scale 字节数为
`slots*(3*H*I + 3*Sr*Sc*sizeof(float))`。两者只统计权重和 scale，
不包含 Router、共享分支和激活。CPU 计算不保留 FP32 权重工作区。
按 `H=2048,I=512,E=256` 计算，一层 RAM routed 参数约 768 MiB（另加 scale），
每个 GPU FP8 专家占 3 MiB + 768 字节 scale。CPU 点积内部解码省去
原来的 12 MiB 权重展开工作区，两个临时激活向量共 4 KiB。
40 层各缓存两个专家共占 240 MiB + 60 KiB scale，不含激活工作区。

### Qwen3.5 文本模型与加载

`src/models/qwen35.hh/.cc` 按 Qwen3 的结构提供 `Qwen35Attention`、
`Qwen35DecoderLayer`、`Qwen35Model` 和继承 `Model` 的 `Qwen35ForCausalLM`。
每层根据 `ModelConfig::layer_types` 构造 full attention 或 GatedDeltaNet，
FFN 使用带共享专家的 MoE。`forward()` 返回最终归一化的 hidden states，
`compute_logits()` 单独调用 LM head；prefill 只取每条序列的最后一行。

```cpp
#include "models/qwen35.hh"

const auto model_dir = std::filesystem::path("/path/to/Qwen3.5-35B-A3B-FP8");
const auto config = ModelConfig::read_from_file(model_dir);
Qwen35ForCausalLM model(config, Device::cuda(0),
                        std::vector<std::int32_t>{0, 2});
const auto loaded = model.load_ckpt(model_dir);
Qwen35Cache cache(config, /*num_blocks=*/128, /*block_size=*/16,
                  /*state_slots=*/4);
model.transformer().set_cache(cache);
// Prepare GPU Int32 input_ids/positions and a ForwardContext, then:
// auto hidden = model.forward(input_ids, positions, context);
// auto logits = model.compute_logits(hidden.view().as_const(), context);
```

这个例子展示直接调用模型的接口；Engine/CLI 负责准备调度元数据和缓存。GPU
expert ID 名单应用于每层；默认名单为空，仅把 routed 权重放在 RAM。CPU 构造可以检查和加载参数，
forward 需要 CUDA 设备 0。所有参数需完整加载后才能执行。

配置按 `architectures` 读取：ConditionalGeneration 使用 `text_config` 与
`model.language_model.*` 权重前缀，CausalLM 使用文本配置与 `model.*` 前缀。
只实现文本推理，视觉模块及 MTP 权重不参与加载。其他未使用的文本 tensor 默认
报错，可用 `ModelLoadOptions::allow_unexpected_tensors` 放宽。

模型加载器直接读取 safetensors 分片，按实际 dtype 转换。Routed FP8 字节原样
复制到 RAM，BF16/F32 scale 转成 FP32；其他 BF16/F32 参数转成 FP32，非专家
FP8 矩阵乘对应的 `128x128` scale 后展开到模型设备。GPU routed 专家复制
FP8 权重和 FP32 scale，在 GEMM 内融合权重反量化；输入、乘法、累加
和输出都保持 FP32。非专家模块与 CPU 未命中专家也使用 FP32 激活。

加载先验证全部所需文本 tensor 的名称、shape、dtype 和 scale 网格，再修改参数。
元数据错误保留原模型；开始复制后若失败，该模型不能再执行，需重新成功加载。
成功加载自动填充各层 GPU 专家缓存。重新加载会解绑旧的 KV/递推缓存，之后需
重新绑定已重置的缓存。加载、forward、缓存重置与填充需串行。

`loaded.num_tensors/num_bytes` 统计唯一源 tensor 和源字节数，包含反量化使用的
scale，和运行时 RAM/VRAM 占用不同。原先公开的单模块/单层加载接口已移除，
统一使用模型的 `load_ckpt()`。

### 两种 attention 与执行缓存

Full gated attention 的计算顺序是：Q 投影按 head 拆成 `[query, gate]`，
Q/K 分别做 zero-centered RMSNorm，施加 partial RoPE，执行 causal GQA，
将结果逐元素乘 `sigmoid(gate)`，最后做输出投影。Prefill 支持整段或带已有前缀
的分块输入，decode 使用现有 paged attention kernel。

`src/layers/gated_delta_net.hh/.cc` 的顺序是 QKV 投影、depthwise causal
卷积和 SiLU、Q/K L2 normalization、逐 token 更新状态矩阵、gated RMSNorm、
输出投影。Q/K head 按组重复以匹配 value head；每个 value head 有自己的状态：

```text
beta = sigmoid(b)
g = -exp(A_log) * softplus(a + dt_bias)
S = exp(g) * S
delta = beta * (v - k^T S)
S = S + k delta^T
output = (q / sqrt(key_head_dim))^T S
```

这里的 q/k 已经做 L2 normalization。Prefill 和 decode 共用顺序递推 kernel，
没有实现并行 chunk 算法；目的是让状态更新易于阅读和验证。

`Qwen35Cache` 分开持有两类状态，不注册为模型参数：

- Full 层的 K/V：`[full_layers, num_blocks, block_size, kv_heads, head_dim]`。
- Linear 层的卷积历史：`[linear_layers, state_slots, conv_dim, kernel_size-1]`；
  递推矩阵：`[linear_layers, state_slots, value_heads, key_dim, value_dim]`。

缓存借用视图，必须比绑定它的 forward 活得更久。Full attention 的 prefill 和
decode 都调用 `flash_attn_with_kvcache`，在同一个 kernel 中写入本轮 KV 并计算 attention。
`ForwardContext` 通过 `block_tables`、`cu_seqlens_q/k` 和 `max_seqlen_q/k`
描述变长 Q 和 paged KV；`cu_seqlens_k` 对应的长度包含历史和当前 token。
可选的 GPU Int32 `[B]` `state_indices` 将本次 batch 行映射到序列固定的递推状态槽位。
batch 换序时更新映射；空视图表示槽位等于 batch 行号，同一次调用不得重复槽位。
decode 每条序列输入一个 token，`cu_seqlens_q` 为 `[0, 1, ..., B]`。

新序列复用槽位前调用 `cache.reset_state(slot, stream)`。它清空该槽位所有
linear 层的历史和状态，旧 K/V 的可见范围则由 block table 与长度控制。
重置与 forward 应使用同一 stream，跨 stream 前需同步。Full attention 始终需要
绑定 KV cache；GatedDeltaNet 没有绑定状态缓存时只支持从零状态开始的 stateless
prefill。现有 prefix cache 只保存 K/V 块，不能复用 GatedDeltaNet 的递推状态，
因此 Engine 已对 Qwen3.5 关闭 prefix reuse。

按 35B-A3B 的 10 个 full 层和 30 个 linear 层计算，当前 FP32 K/V 每个逻辑
token 占 40 KiB；linear 状态每个序列约 62.8 MiB，与上下文长度无关。
非专家权重展开和全部 FP8 routed 专家仍需足够的 VRAM/RAM；本次验证环境的
WSL 只分到 22 GiB RAM，未加载完整 35B 模型。以后运行整模型前需确认 WSL
确实能使用机器的 48 GiB 内存。

### Engine 中的序列状态槽位

`ModelRunner` 持有 `std::unique_ptr<Model>`，通过虚函数调用 `forward()` 和
`compute_logits()`。Qwen3 绑定 `Qwen3Cache`，分配全部层的 K/V；Qwen3.5
绑定 `Qwen35Cache`，只给 full 层分配 K/V，并为 `max_num_seqs` 条序列预分配
linear 状态。

`Scheduler` 在 Qwen3.5 模式维护空闲槽位队列。`Sequence::state_slot` 的生命周期是：

1. 首次分配 KV 块时取一个空闲槽位；没有槽位时等待，让活跃序列继续 decode。
2. 分块 prefill 和 decode 保留同一槽位。ModelRunner 按本轮 batch 顺序上传
   `state_indices`，槽位不随 batch 行号变化。
3. 结束、抢占或 `clear()` 时归还槽位。抢占也释放 KV，保留 token IDs，之后
   从 token 0 重算。首次/重算 prefill 前在推理 stream 上清零槽位的全部状态。

异常路径会清理本轮请求；后续新请求拿到旧槽位时仍会清零。KV 前缀复用关闭，
避免复用只有 K/V 而缺失递推状态的上下文。Qwen3 使用原有前缀缓存和调度策略。

### Qwen3.5 基础算子和非专家加载

`RMSNorm` 增加末尾的 `zero_centered` 构造参数，默认 `false` 保持 Qwen3 的
语义；设为 `true` 时，在 FP32 中完成归一化并乘 `1 + weight`，最后转换到
输出 dtype。带 residual 的版本先将相加结果舍入到 tensor dtype，再归一化。

新增 `layers/gated_rms_norm.hh/.cc`，供 GatedDeltaNet 使用。它直接乘 weight，
再逐元素乘 `SiLU(gate)`，目前只支持 FP32 计算。两种 norm 不可互换：

```text
zero-centered: y = x / sqrt(mean(x²) + eps) * (1 + weight)
gated:         y = x / sqrt(mean(x²) + eps) * weight * SiLU(gate)
```

`RotaryEmbedding` 增加末尾的 `rotary_dim` 参数，默认 `0` 表示旋转整个 head。
35B-A3B 的 head_dim=256，partial_rotary_factor=0.25，实际旋转 64 维，
配对 `(i,i+32)`；后面的 192 维保持原值。当前只实现文本位置，配置中的
`mrope_section/mrope_interleaved` 已保留，尚不处理图像/视频的三轴位置。

这些算子注册的参数由 `Qwen35ForCausalLM::load_ckpt()` 统一加载，不再
公开单独的 Qwen3.5 模块加载函数。

### 可选共享专家

`MoEConfig::shared_expert_intermediate_size` 默认是 `0`，关闭共享分支。
设为正数时，增加一个 SwiGLUMLP 和无 bias 的 `H → 1` Linear：

```text
y = routed_sum + sigmoid(shared_expert_gate(x)) * shared_expert(x)
```

共享专家对整批 token 执行，每个 token 使用自己的门控，不参与 Router 的
top-k，也不参与选中专家概率的归一化。它的中间维度可以与 routed 专家不同。
门控 logit 为 `0` 时，sigmoid
系数是 `0.5`；关闭分支应将配置维度设为 `0`。

```cpp
MoEConfig config{
    .hidden_size = 128,
    .intermediate_size = 64,
    .num_experts = 16,
    .top_k = 2,
    .shared_expert_intermediate_size = 32,
};
ThreadPool cpu_pool(8);
MoE moe(config, cpu_pool, Device::cuda(0), std::vector<std::int32_t>{0, 2, 4, 6});
```

共享分支直接挂在 `MoE` 下，注册以下参数：

```text
shared_expert.gate_proj.weight   [Is,H]
shared_expert.up_proj.weight     [Is,H]
shared_expert.down_proj.weight   [H,Is]
shared_expert_gate.weight        [1,H]
```

GPU 构造时这四个参数常驻 GPU，不占 routed expert cache 的槽位；CPU 构造时
它们位于 RAM，供独立参考实现使用。共享分支额外的 FP32 权重字节数是
`(3 * H * Is + H) * sizeof(float)`，不含激活和临时工作区。初始化参数仍然通过
`named_parameters()`；`populate_expert_cache()` 只上传 routed 专家的缓存副本。

这一版先完成 routed dispatch 和 combine，再用现有层计算共享专家、门控，并
通过 `kernels::cuda::moe_add_shared()` 原地加入输出。共享分支尚未加入
CPU/GPU 计算重叠，工作区使用现有层的临时分配。

`MoEOutput::shared_expert_output` 返回未经门控的 `[T,H]` 输出，
`shared_gate_logits` 返回 sigmoid 之前的 `[T,1]` logits；关闭分支时两者为空。
最终 `hidden_states` 包含 routed 和共享两部分。

### CPU/GPU 重叠执行

`ExpertExecutor::forward_into()` 统一使用 CPU/GPU 重叠执行，无需选择执行模式：

```cpp
ThreadPool cpu_pool(8);
MoE moe({128, 64, 16, 2}, cpu_pool, Device::cuda(0),
        std::vector<std::int32_t>{0, 2, 4, 6});
```

executor 将 `[T,K]` 的分派按专家计数分组，先提交 CPU 专家任务，再提交 GPU
专家计算，因此两者可以重叠。GPU 专家、CPU 输出上传、scatter 和 combine
使用同一个 stream 排序。executor 返回前完成全部 CPU 任务，GPU 工作可以仍在
排队。`MoE::forward()` 的单 token 路径把路由 ID 与输入一起下载，只等待下载
完成事件；共享专家提前提交，与 CPU 专家重叠。层末记录完成事件后异步返回，
下一层通过同一 stream 排序与下载事件安全复用工作区。工作区切换到另一 stream
或析构时等待先前使用完成；profiling 保留层末等待以维持计时口径。
内部 executor 的 cache 就绪状态、张量形状及专家编号是调用方的前置条件，
Debug 构建通过断言检查；Release 测试只传入满足这些条件的分派。

`FP8Expert::forward_into()` 使用预先分配的 gate/up/activated 工作区，直接
写入输出行。所有 GPU 专家共用这三个缓冲区，依靠同一 stream 排序；提交 GPU
专家时不会分配或释放 CUDA Tensor。CPU 输入与输出暂存区也会复用，输出区
按需要增长。返回的输出拥有独立存储，后续调用不会覆盖先前结果。

`Qwen35Model` 持有一份 `MoEExecutionWorkspace`，各层共享，避免每层长期保留
一套 `[T*K,H]` 缓冲。单独构造 `MoE` 时默认拥有自己的工作区，也可通过第五个
构造参数传入外部工作区。批量传输暂存区使用 pinned RAM，CPU 分组输入使用普通
RAM；单 token 输入/输出保留小规模普通 RAM 暂存路径。GPU 分组输入、分组输出、
排列映射和专家中间缓冲均按容量复用。扩容只在
提交依赖这些缓冲的任务之前发生。工作区及同一模型的调用必须串行。

这一版重叠 CPU/GPU 计算，输入下载仍在计算之前完成，CPU 输出在计算完成后
上传；没有引入多 stream 传输流水线。线程数由 `--cpu-threads` 控制，默认 8。
在未命中的多 token 场景，整层 MoE 的显式 stream 同步从约 `2*T+2` 次减少到
3 次；显存分配器和其他模块可能还有同步，不能据此直接推导耗时缩减比例。

三个辅助类与 `MoE` 一起放在 `src/layers/moe.hh/.cc`。建议在 `.cc` 中按
`ExpertStore` 的 RAM 权重 → `ExpertCache` 的固定 GPU 槽位 →
`ExpertExecutor` 的分派 → `MoE::forward` 的路由与合并阅读。运行时 CPU 专家
使用 `kernels::cpu::expert_forward()` 和 FP32 累加；仅供旧测试使用的完整 CPU
MoE 对照实现已移除。

## MoE 分组执行验证

使用合成权重，不依赖模型下载，需要 CUDA GPU：

```bash
cmake -S . -B build -DNANO_VLLM_CPP_BUILD_TESTS=ON
cmake --build build --target nano-vllm-cpp-moe-batch-test -j4
ctest --test-dir build -R '^moe-batch$' --output-on-failure
```

独立标量 E4M3FN 解码和 FP64 点积参考覆盖全 CPU、混合和全 GPU 驻留、单 token
快速路径、多行 GEMM、128 scale 边界、非整块尺寸、空输入、重复/偏斜路由、
工作区扩容后缩小、跨层共享、输出所有权、assignment 计数和共享专家 combine。
GPU GEMM 与 GEMV 的累加顺序不同，数值按容差对齐。

该程序也提供单层 MoE 微基准：

```bash
./build/nano-vllm-cpp-moe-batch-test --benchmark
```

它使用 `H=2048,I=512,E=16,K=8`、3 个 GPU 常驻专家和 8 个 CPU worker，分别
测量 `T=1/128`；T=1 预热 10 轮、统计 51 轮，T=128 预热 2 轮、统计 9 轮，
报告中位数与最小/最大值。它不代表完整 35B
模型性能；整模型比较请继续使用上面的 `nano-vllm-cpp-bench` 和相同配置。

## SIMD 改动之前的 Qwen3.5 验证

CPU 点积内部解码在 Ryzen 7 9700X 上验证：256 种 FP8 编码与 CUDA SDK
转换一致，24 次专家计算覆盖非整块尺寸、零输入及官方 35B 的一个真实专家，
与旧的先展开再计算路径逐位一致（最大绝对误差 0）。RTX 5060 Ti 上空、
混合、全部 GPU 驻留的 9 次 executor 和 9 次带共享专家的 MoE 调用与修改前
逐位一致，CUDA memcheck 为 0 个错误。

当前融合权重反量化的 FP32 GEMM 在 RTX 5060 Ti、CUDA 13.3 上验证：
462 组用例覆盖未补齐的 M/N/K、未对齐地址、全部 256 种 FP8 编码、128
scale 边界、零输入、空维度，以及避免激活量化误差的小差值用例。
对照独立 FP32 权重反量化参考和 cuBLAS FP32 GEMM；有限结果的最大
`abs(error)/(1+sum(abs(products)))` 为 `4.92e-7`。CUDA memcheck、
racecheck 和 synccheck 均为 0 个错误。

此前 cuBLASLt 原生 GPU FP8 GEMM 的验证记录（含激活动态量化）：覆盖 256 种 FP8
编码、11 组矩阵尺寸（含 padding、零输入和空 batch），并加载官方 35B 第 0 层
的一个真实专家。对照独立的 CPU 动态激活量化参考，Linear 最大绝对误差为
`1.91e-6`，真实专家为 `1.75e-10`；CUDA memcheck 为 0 个错误。另检查了
空、混合、全部 GPU 驻留名单下的 9 次 executor 与 9 次带共享专家的 MoE
调用，GPU 分支使用量化参考，CPU 分支使用 FP32 参考。未验证完整 35B 生成。

下面记录的是接入 cuBLASLt 原生 GPU FP8 GEMM 之前的验证。

验证运行在 RTX 5060 Ti、CUDA 13.3，数值参考使用 Transformers 5.15.1 的
官方 Qwen3.5 MoE FP32 实现，并读取相同的 BF16/FP8 权重后展开。临时验证程序
放在 `/tmp/nanovllm-qwen35-forward` 与 `/tmp/nanovllm-qwen35-engine`，未重新
添加已删除的 tests 目录或 CTest 项。

- 四层小模型（3 个 linear + 1 个 full，4 个 routed 专家，top-2 与共享专家）
  的 hidden states、logits、卷积历史和递推状态对齐，最大绝对误差小于 `2e-6`。
  覆盖 packed 多序列、分块 prefill、decode、batch 换序、槽位复用和 71-token
  prefill，以及 CPU misses、GPU hits、混合驻留和两种执行模式。
- 官方 35B FP8 第 0 层 GatedDeltaNet 与第 3 层 full attention，对齐整段
  prefill、带前缀的分块 prefill、decode 及最终缓存，最大绝对误差小于 `4e-6`。
- 完整小模型的 CUDA memcheck 与 racecheck 均通过；原有 Qwen3-0.6B 的
  `Hello`、贪心生成 8 token 与此前结果一致。

Engine/CLI 接入后，额外验证了六条请求共享两个状态槽位、4-token 分块 prefill、
三轮重复调用，以及 8 个 KV 块触发的抢占重算。所有生成 token 与独立的官方
FP32 全上下文推理一致；覆盖空/混合 GPU 专家名单及两种 checkpoint 前缀。
官方 35B tokenizer/聊天模板的四种纯文本输入与 Transformers 完全一致，
Qwen3-0.6B 重复生成保留原有结果。Engine 调度的 CUDA memcheck 为 0 个错误，
Scheduler 的 ASan/UBSan 生命周期压测通过。

这些检查未验证完整 35B 文本生成和视觉输入，也没有宣称与官方动态 FP8 推理
逐 token 一致或报告整模型速度。
