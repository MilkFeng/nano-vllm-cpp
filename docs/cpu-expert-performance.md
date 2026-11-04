# CPU FP8 专家优化

2026-11-04，Ryzen 7 9700X（8 核 / 16 线程、32 MiB L3）、RTX 5060 Ti
16 GiB、WSL2 Linux、GCC Release / CUDA 13.3。模型为本机
Qwen3.5-35B-A3B-FP8，专家 H=2048、I=512、FP8 E4M3FN、128×128 分块缩放。

## 实现

- AVX2 解码由 gather 查表改为指数、尾数及符号的位运算，显式处理非正规数、
  正负零和 NaN；多 token prefill 也复用该解码。
- 单 token 新增 AVX-512 路径，Gate/Up 同时处理 4 个输出通道并计算 SiLU×Up，
  Down 同时处理 8 个输出通道。AVX2 保留融合 Gate/Up 和 2 通道 Down。
  保持 FP32 权重缩放、FMA 累加和原来的两条 8 lane 归约顺序。
- AVX-512 只对单独的源文件启用，运行时检查 CPU 和 OS 支持；其他代码仍按
  AVX2 编译。默认自动选择，可用 `NANOVLLM_CPU_ISA=avx2` 或 `avx512` 强制对照，
  该变量在第一次调用时读取。不支持 AVX-512 时强制选择会报错。
- 每层 CPU 专家统一提交：一次加锁入队、共享完成计数和异常状态，替代每专家一个
  `std::function` / `packaged_task` / future。较大批次用一次 `notify_all` 唤醒。
  top-8 的任务上下文使用栈数组；批次状态和描述符仍有分配，尚未声称完全零分配。
  prefill 同样采用批量提交。异常路径等待所有 CPU 任务完成后再释放缓冲区。
- 新增 `--cpu-affinity`（默认关闭）。读取当前允许的 CPU 拓扑，每物理核选一个
  逻辑 CPU，worker 不使用同核的 SMT 兄弟。有空余核时，把模型 forward 的调用线程
  留在第一个物理核，worker 从第二个核开始；占满物理核时调用线程与首个 worker
  共用第一个逻辑 CPU。退出 forward 后恢复原调用线程亲和性。

## 整模型结果

基线为修改前冻结的源码及静态库，和优化版分别构建带相同计时点的测量程序。
每个缓存配置单独加载模型，预热 1 轮，测量 3 轮；B=1，输入/输出各 128 token，
固定 prompt、seed=42、temperature=0、忽略 EOS、关闭前缀复用。
GPU 初始专家 ID 为 0,2,4,…，每 4 个 decode step 自适应换入专家，最多换入 8 个。
默认 8 worker、不绑核。以下吞吐为三轮 tok/s 的算术平均，CPU 等待仅统计
MoE 在提交 GPU 工作后等待 CPU 批次完成的时间，不包含 CPU 提交本身。

| 每层 GPU 缓存 | 修改前 tok/s | 修改后 tok/s | 提升 | CPU 等待 ms/token，前 → 后 |
|---|---:|---:|---:|---:|
| 3 | 10.25 | 14.23 | 38.8% | 43.30 → 15.15 |
| 16 | 10.80 | 13.95 | 29.2% | 36.34 → 12.83 |

原始吞吐：

- 缓存 3：前 9.36 / 10.41 / 10.97，后 14.02 / 14.31 / 14.36。
- 缓存 16：前 10.70 / 10.72 / 10.97，后 13.81 / 14.29 / 13.74。

六轮各 128 个输出 token 与对应基线完全一致，专家分配与替换统计也一致。
两组峰值 Tensor 显存分别约 10.24 / 11.77 GiB，未因本次 CPU 优化增加。
上述是 Linux 本机实验，不能直接与 Windows / CUDA 12.8 的旧测量做绝对吞吐比较。
本次实现仍有 GPU kernel、提交、同步与其他层计算成本，因此专家微基准加速比
不能直接当作整模型加速比。

## CPU 微基准与线程分配

使用模型第 0 层的 64 个真实专家，合计约 192 MiB 权重。轮换步长为当批任务数，
因此 1/2/4/8 个并行任务都会遍历全部 64 个专家，工作集超过 L3；另测固定同一专家
的热缓存情形。以下是轮换情形，100 次测量的批次完成时间中位数，包含线程池
提交与等待，不是仅计算内核时间。冻结的旧实现和优化版交替运行，以减少温度、
频率变化的偏差。微基准的旧实现使用逐任务 future，优化版使用批量提交。

| 默认 8 worker、不绑核 | 原 AVX2 | 优化 AVX2 | 优化 AVX-512 |
|---|---:|---:|---:|
| 单专家 µs | 约 756 | 411 | 290 |
| 8 专家一批 µs | 约 1400 | 775 | 589 |

AVX-512 的 8 专家批次，不同 worker 配置：

| worker | 不绑核 µs | 绑物理核 µs |
|---|---:|---:|
| 6 | 640 | 652 |
| 7 | 676 | 665 |
| 8 | 589 | 557 |

当前机器允许的逻辑 CPU 为 0–15，同核的 SMT 配对为 (0,1)、(2,3)、…、(14,15)。
绑定时，6/7 worker 使用 CPU 2,4,…,12/14，调用线程使用 CPU 0；8 worker 使用
CPU 0,2,…,14，调用线程与首个 worker 共用 CPU 0。
8 个空任务的提交及等待往返，不绑核时 8 worker 从 161 µs 降至 151 µs，
6 worker 从 168 µs 降至 122 µs。这里包含线程唤醒和主线程等待，不能把它当作
纯入队耗时，也不能把全部专家加速归因于调度。

整模型另外测了缓存 16、7 worker、绑物理核并预留提交核的组合，条件与上表相同。
三轮为 14.41 / 14.38 / 14.68 tok/s，均值 **14.49 tok/s**，相对默认 8 worker、
不绑核的 13.95 提升约 **3.9%**，CPU 等待约 11.89 ms/token。三轮输出 token、
专家分配与替换统计均与对应基线一致。该组合在本机略有收益，可以使用：

```bash
./build/nano-vllm-cpp-bench --model /path/to/Qwen3.5-35B-A3B-FP8 \
  --input-tokens 128 --output-tokens 128 --batch-sizes 1 \
  --num-blocks 32 --block-size 16 --max-batched-tokens 128 \
  --seed 42 \
  --cpu-threads 7 --cpu-affinity \
  --gpu-expert-ids 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 \
  --expert-adapt-every 4 --warmup 1 --repeat 3 --output cpu-expert.json
```

这一组测试同时改变了 worker 数与绑定策略，不能单独归因于绑核。
微基准覆盖全部 6/7/8 × 绑定/不绑定组合，整模型只比较默认 8、不绑定与 7、绑定；
其他 CPU、NUMA、SMT、推理并发度和 prompt 的最优分配仍可能不同。
收益较小且只有单机单 prompt，因此保留默认 8 worker、不绑核，亲和性由参数开启。

## 验证与复现

```bash
cmake -S . -B build -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
cmake --build build --target nano-vllm-cpp-cpu-check nano-vllm-cpp-cpu-bench -j 8
NANOVLLM_CPU_ISA=avx2 ./build/nano-vllm-cpp-cpu-check
NANOVLLM_CPU_ISA=avx512 ./build/nano-vllm-cpp-cpu-check
NANOVLLM_CPU_ISA=avx512 ./build/nano-vllm-cpp-cpu-bench \
  /path/to/Qwen3.5-35B-A3B-FP8 cpu-avx512.json 100
```

CPU 检查覆盖所有 256 种 FP8 编码、不同规模和缩放块的尾部、多输出通道跨缩放行、
完整单/多 token 专家、混合单任务与批次、并发提交、异常、移动与析构等待、亲和性恢复。
完整分块的单 token 主路径要求逐位一致；任意尾部对独立标量参考使用误差界，
因为编译选项及多 token 循环的标量浮点收缩可能不同。
额外使用冻结旧实现做 Release 差分检查，覆盖尾部尺寸及 1/2/3/4/7 token，
以及 64 个真实专家的单 token 输出，均逐位一致。
AVX2 / AVX-512 的 AddressSanitizer、UndefinedBehaviorSanitizer 和 LeakSanitizer
均通过；没有依靠降低模型精度获得加速。Windows 亲和性与 ISA 检测已实现，
本环境未做 Windows 编译或性能测试。

原始 JSON、各轮日志、冻结参考实现、测量程序、源码与可执行文件哈希位于
`profile/cpu-experts-20261104/`。该目录不提交二进制和模型权重。
