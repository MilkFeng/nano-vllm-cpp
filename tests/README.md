# 小尺寸 MoE 验证

配置构建时开启 `-DBUILD_TESTING=ON`，然后运行：

```bash
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

`moe_cpu` 不需要 GPU，验证手工可计算的两维示例、路由权重归一化和配置限制。
`moe_cuda` 在 CUDA device 0 上运行，缺少 CUDA 设备或驱动时标记为跳过。

GPU 测试使用非默认、nonblocking stream，对比 CPU 参考实现的 Router logits、
专家编号、路由权重、每个专家的原始输出和最终合并结果。覆盖默认的
`H=128, I=64, E=16, K=2`，以及 top-1、选择全部专家、单专家、非整齐维度、
相同分数、大幅度分数和重复执行。CPU 参考使用 FP32 权重及中间结果，点积
使用 double 累加；比较使用绝对误差与相对误差容限。

单独运行可看到设备名称和每个 MoE 用例的最大输出绝对误差：

```bash
./build/nano-vllm-cpp-moe-test --cpu
./build/nano-vllm-cpp-moe-test --gpu
```

检查 CUDA 内存访问和同步：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
  ./build/nano-vllm-cpp-moe-test --gpu
compute-sanitizer --tool racecheck --error-exitcode 1 \
  ./build/nano-vllm-cpp-moe-test --gpu
```

这些测试使用固定的随机权重或手工构造的权重，验证一个独立 MoE 层的数学
流程；完整模型的文本生成仍走现有 Qwen3 链路。

## 固定专家缓存与 CPU/GPU 分担

`expert_cpu` 不需要 GPU，检查实际 FP32 CPU 专家计算与独立 double 累加参考
实现的误差，以及 RAM 权重存放、空缓存和无效专家名单。
`moe_offload_cuda` 使用非默认 nonblocking stream 验证全 GPU、全 CPU、
4 槽位混合执行、4 槽位全未命中、4 槽位全命中。两组输入分别选择
`[7,2]` 和 `[9,4]`；改变路由后再次执行原输入，检查输出行和临时缓冲区。

测试检查缓存只分配指定数量的 GPU 专家，全部 routed 专家参数仍在 RAM，缓存
副本没有额外注册为模型参数。还检查填充完成前禁止使用缓存、GPU 副本独立
于 RAM 参数，以及显式重新填充后的结果。输出记录每次的专家编号、CPU/GPU
执行计数、槽位数量、GPU 权重字节数和最终输出误差。

```bash
./build/nano-vllm-cpp-moe-offload-test --cpu
./build/nano-vllm-cpp-moe-offload-test --gpu
compute-sanitizer --tool memcheck --error-exitcode 1 \
  ./build/nano-vllm-cpp-moe-offload-test --gpu
```

## 共享专家

`moe_cpu` 增加可手算的共享分支：输入 `x=[1,2]`，记 `s=SiLU(1)`，
routed 输出是 `[2s,7s]`，原始 shared 输出是 `[4s,-8s]`。共享门控 logit
为 `0` 时，sigmoid 系数为 `0.5`，最终结果应是 `[4s,3s]`。测试还检查
`-2`、`2`、`-1000`、`1000` 的门控，以及无效共享矩阵形状；共享分支不能
改变 Router logits、专家编号、路由权重和 routed expert 原始输出。

`moe_cuda` 对同步和 Overlap 模式都比较共享专家原始输出、门控 logit 与最终
结果，覆盖共享/routed 中间维度不同、top-1、top-E、单专家和非整齐维度。
共享合并算子单独用 `H=513` 检查多 CUDA block、极端 sigmoid 值、原地加法
和只读输入。关闭共享分支的原有测试继续运行，返回的共享中间结果应为空。

`moe_offload_cuda` 在 `H=128, I=64, Is=37, E=16, K=2` 上检查全 GPU routed、
全 CPU routed 和混合缓存。共享专家和门控始终在 GPU，另外注册四个参数；
测试检查 GPU 共享权重字节数、routed 缓存大小和 routed CPU/GPU 计数，并要求
同步/Overlap 的各项结果逐值相同。所有数值结果另外与独立 CPU 参考对照。
上述 CTest 和 compute-sanitizer 命令同时覆盖共享分支。

## CPU/GPU 重叠执行

现有 `moe_cuda` 对同步版和重叠版都检查 top-1、top-E、单专家、非整齐维度、
随机权重和重复执行。`moe_offload_cuda` 对两版直接比较 Router logits、专家
编号、路由权重、未加权专家输出、合并结果和 CPU/GPU 数量，要求两版逐值相同；
两版还分别与独立 CPU 参考实现比较，使用浮点误差容限。

重叠测试覆盖缓存未填充、刷新 RAM 权重后重新填充，以及同一个 Executor 改变
K、输入、专家顺序和 CUDA stream。保留第一次返回的输出，后续调用和工作区
增长完成后再次检查它，验证返回值不与内部暂存区共享存储。

手工测量使用 `H=4096, I=2048, E=8, K=8`，固定 4 个 GPU 专家、4 个 CPU
专家，RAM 专家权重 768 MiB、GPU 专家权重 384 MiB。预热两轮后交替测量
7 对同步/重叠 dispatch，报告各自中位数。这里只测一个合成 MoE dispatch，
不包含 Router/combine，也不能代表完整模型速度。

```bash
./build/nano-vllm-cpp-moe-offload-test --benchmark
```

用 Nsight Systems 捕获一轮预热后的重叠执行，CPU 采样关闭以便在 WSL 上运行：

```bash
nsys profile --trace=cuda,cublas --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /tmp/moe-overlap ./build/nano-vllm-cpp-moe-offload-test --profile-overlap
```

时间线中最后一个 GPU 专家的 cuBLAS 提交返回后，到第一次 CPU 输出 H2D
复制提交之前，调用线程正在计算 CPU 专家。检查 GPU kernel 是否与这个区间
相交，并检查其中没有 `cudaMalloc`、`cudaFree` 或 stream/device 同步。
替换为 `--profile-sync`、换一个输出文件名，可以捕获原来的分派作为对照。
`.nsys-rep` 可以在 Nsight Systems 中打开；`nsys export --type=sqlite` 可以
导出时间戳做交集计算。普通 `--gpu` 测试不要求计时更快，避免把噪声当成失败。

2026-10-11 本机测量：AMD Ryzen 7 9700X、RTX 5060 Ti 16 GB、WSL Linux、
CUDA 13.3、Release 构建，使用上述合成配置，CPU 专家在调用线程使用标量 FP32。
7 对交替测量的 dispatch 中位数：同步 37.88 ms，重叠 36.49 ms。Nsight Systems
2026.1 捕获的另一次单次 dispatch 中，最后一个 cuBLAS 提交返回到第一次 CPU
输出上传提交的区间为 36.16 ms；16 个 GPU kernel 与该区间相交，交集累计
0.894 ms。GPU 提交和 CPU 计算阶段没有 CUDA 分配、释放或同步调用。
这些数字来自合成测试；完整 Qwen3 文本生成仍使用原路径。
