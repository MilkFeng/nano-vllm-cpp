# FP32 GEMM：可复现测试与简历指标

这里包含所有 GEMM 实验代码。生产实现仍在 `src/kernels/cuda/linear.cu`，
直接计算 `X[M,K] @ W[N,K]^T`，支持可选 bias、尾块及非对齐连续视图。

已完成的 RTX 5060 Ti 测量见 [完整报告](results/rtx5060ti-20261030/report.md)，
同目录保存三轮原始数据、汇总、环境与 SHA256，以及 memcheck/racecheck 日志。
33 种尺寸、每项 15 组样本、3 个独立进程：hot 的几何平均为分块基线的
2.63 倍、严格 FP32 cuBLAS 对照的 70.2%，峰值 9.85 TFLOPS；
rotating 分别为 2.15 倍、64.2%、9.71 TFLOPS。
当前不能据此宣称整体优于 cuBLAS，或将算子指标当作整模型 decode 提升。

## 一次完整测试

先按项目说明完成 Release 配置，然后显式构建实验目标：

```bash
cmake --build build --target nano-vllm-cpp-comp-gemm --config Release -j 4
python3 exp/gemm/run_gemm.py --suite full --runs 3
```

Windows PowerShell，对应 `scripts/build.ps1` 的构建目录：

```powershell
cmake --build build/windows/Release --target nano-vllm-cpp-comp-gemm --config Release --parallel
python exp/gemm/run_gemm.py --executable build/windows/Release/nano-vllm-cpp-comp-gemm.exe --suite full --runs 3
```

脚本只需要 Python 标准库。它启动 3 个独立进程，各用一个 seed，默认每项采集
15 组样本，生成 `build/gemm-results/<UTC时间>/report.md`，以及原始 JSON、
逐进程日志、汇总 CSV 和环境/命令/源码/二进制 SHA256。输出目录也可用
`--output-dir <新目录>` 指定。运行时避免其他 GPU 任务；完整测试保留所有结果。

`report.md` 包含完整逐 shape 数据、几何平均、进程间范围及限定测试范围的简历表述。
如果结果没有超过 cuBLAS，写“达到严格 FP32 cuBLAS 对照吞吐的 Y%”，
不要把少数获胜尺寸扩展为整体收益。峰值与几何平均分别注明。

## 预先固定的测试范围

| 分组 | shape，按 M/N/K 排列 | 数量 |
| --- | --- | ---: |
| 方阵 | 256/512/1024/2048/4096 的立方组合 | 5 |
| 矩形 | 128×2048×2048、128×8192×2048、512×2048×2048、128×2048×8192 | 4 |
| Qwen3 Linear | M=1/8/32/128/512；N/K=3072/1024、1024/3072、2048/1024、1024/2048 | 20 |
| 尾块 | 129×513×257、127×2049×1023、255×1001×769、33×129×15 | 4 |

共 33 个 shape，每个测试 hot 和 rotating 两种条件，即每进程 66 项。
Qwen3 尺寸来自 Qwen3-0.6B 的 hidden=1024、intermediate=3072、16 个 query heads、
head_dim=128，覆盖 gate/up、down、q 和 o 投影。它们不是按模型调用频率加权的
端到端吞吐估计，也没有覆盖整个模型的所有算子与 lm_head。

hot 重复使用一组矩阵，并不保证矩阵都能放入 L2。rotating 在同一 GPU 上轮换
最多 256 组输入、权重、输出地址，目标工作集为 `max(64 MiB, 2×L2)`，不包含
workspace。很小的 shape 可能达到槽位上限仍低于该目标，报告会标 `†`。
这种条件模拟地址轮换，不宣称每次访问都为冷缓存。

## 对照与精度

| 实现 | 说明 |
| --- | --- |
| custom | 生产 `linear()`：寄存器分块、float4、双缓冲、寄存器预取、XOR swizzle；小 M 单独分派，M=1 使用 GEMV |
| tiled_16x16 | 共享内存 16×16 分块、每线程一个累加器，W tile 使用一列 padding；不含 register tiling、双缓冲和向量化 |
| cublas_sgemm | `cublasSgemm`，`CUBLAS_PEDANTIC_MATH`，默认 32 MiB workspace，禁用 atomics |
| cublaslt_tuned | `CUBLAS_COMPUTE_32F_PEDANTIC`，默认 32 MiB workspace，从至多 8 个启发式候选实测选择；校准与正式采样分离 |
| cublaslt_legacy | `comp_gemm.cc` 中保留的原 `cublas_linear()`，默认算法、零 workspace、`CUBLAS_COMPUTE_32F` |

主比较逐 case 取 SGEMM 与调优 Lt 中的较快者，legacy 单列。Lt 候选数、候选校准
原始耗时、选中算法 ID 和实际 workspace 大小均保存在 JSON。若严格 Lt 不支持，
明确记录 unsupported，该项主比较使用 SGEMM。此搜索不是穷举所有 cuBLASLt 配置。

性能测试统一为无 bias 的 FP32 输入/累加/输出、alpha=1、beta=0。
脚本额外设置 `NVIDIA_TF32_OVERRIDE=0`。严格模式及启发式 API 参见
[NVIDIA cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t)。
FP16/BF16 仅做 API 正确性检查；本实验不与 TF32、FP16 或 BF16 Tensor Core 吞吐混比。

## 计时、统计与正确性

输入生成、显存分配、H2D、handle/descriptor 初始化、workspace、算法选择和 Graph
构建都在计时外。使用非默认 non-blocking stream；先预热至少 20 次或一次完整
地址轮换，试跑后选择每批调用次数，使 event 样本目标约 10 ms，最多 1024 次。
批次数必须是地址槽位数的整数倍。实际次数和全部样本都保留，达到上限时批次
可能短于目标时间。

每项用 CUDA Graph 连续执行多次 GEMM，以 CUDA events 测 GPU 时间。
每轮随机交错各实现，shape 及缓存模式顺序也随 seed 打乱，以减少顺序偏差。
它测量 Graph 中的 GPU 执行时间，不包含 CPU 提交成本，不代表完整 Linear 的
主机墙钟耗时或整模型 TPOT。

每进程先取 15 组样本中位数，再对 3 个进程中位数取中位数。逐 shape 的
`baseline_us / custom_us` 表示加速比，跨 shape 用等权几何平均。报告保留全部
尺寸、进程范围与变异系数，范围不是置信区间，不删除慢样本。

每个性能 shape 的每个实现全量输出与严格 SGEMM 比较，并抽取最多 1024 个位置
与独立 FP64 点积比较，要求 `|error| <= 2e-4 + 2e-4*|reference|`。
每进程还先检查 180 组生产 API case 与 30 组分块基线 case，覆盖
FP32/FP16/BF16、bias、M=0/1、尾块和偏移一个元素的连续视图。

## 辅助命令

```bash
# 只检查正确性
./build/nano-vllm-cpp-comp-gemm --check-only

# 小规模冒烟，不作为最终简历证据
python3 exp/gemm/run_gemm.py --suite quick --runs 1 --samples 3 --target-ms 2 --warmup 3

# 单进程完整原始数据
./build/nano-vllm-cpp-comp-gemm --suite full --cache both --output build/gemm-single.json

# 使用已保存的原始数据重新生成报告
python3 exp/gemm/run_gemm.py --report-only build/gemm-results/<目录>

# 检查新基线与生产 kernel
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/nano-vllm-cpp-comp-gemm --check-only
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/nano-vllm-cpp-comp-gemm --check-only

# 汇总逻辑检查：防止算术平均、漏项、错误精度或不一致结果混入报告
python3 -m unittest discover -s exp/gemm -p test_report.py -v
```

`--help` 可查看采样数、warmup、目标时间、workspace 和候选数选项。

## 生产 kernel 的 shared-memory 布局

大 M 保留 128×128 block tile、8×8 register tile、K tile=16。
`2 <= M <= 8` 使用 8×64 tile，`9 <= M < 128` 使用 32×128 tile；M=1 使用 GEMV。
W 沿 `[N,K]` 行内的 K 方向加载，再分量写入 shared memory 的 `[K,N]` tile，
计算阶段沿 N 使用 float4 读取，无需预转置、padding buffers 或临时矩阵。

X 保留原来的行分组 XOR。W 在保持 float4 连续的列置换上增加
`((k / 4) & 3) << 3` 的 XOR，将转置写入时的不同 K 向量映射到不同 bank 组，
同时保留列索引低两位以允许四元素向量读取。尾块补零读取并按边界写回，
非对齐 FP32 地址回退到标量访存。FP16/BF16 以 FP32 shared memory 和累加器计算。



实现 FP32 CUDA GEMM/GEMV，支持 X@Wᵀ，采用寄存器分块、float4 向量化、共享内存双缓冲与 XOR swizzle；在 RTX 5060 Ti、33 种预定义尺寸的同矩阵复用测试中，吞吐几何平均达到严格 FP32 cuBLAS 对照的 70.2%，为 16×16 共享内存分块基线的 2.63 倍，峰值达 9.85 TFLOPS。