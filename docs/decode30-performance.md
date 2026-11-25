# Qwen3.5-35B-A3B-FP8 Decode（2026-11-05）

RTX 5060 Ti 16 GB、Ryzen 7 9700X、WSL2 约 39 GiB RAM、Release / CUDA 13.3。
单序列、输入/输出各 128 token，预热 1 轮；greedy、忽略 EOS、关闭前缀复用。
普通测速正式 3 轮，后续带轻量 CPU 计时的诊断正式 5 轮。

## 结果

**没有完成稳定达到 30 tok/s 的验收。按用户要求，停止后续测速和优化。**

最新一组带轻量计时的诊断均值为 **27.61 tok/s**。此前有一组均值达到
**32.37 tok/s**，但后续独立复核仅 **26.76 tok/s**，不能把最好一组写成稳定结果。

| 阶段 | GPU 专家/层 | CPU worker / affinity | 正式轮次的 Decode tok/s | 算术均值 |
| --- | ---: | --- | --- | ---: |
| 初始基线 | 3 | 8 / 关闭 | 13.34 / 14.46 / 14.66 | 14.15 |
| 紧凑 GPU 权重 | 3 | 8 / 关闭 | 18.36 / 18.53 / 17.48 | 18.13 |
| 分段执行与短时轮询 | 33 | 7 / 开启 | 20.39 / 23.30 / 22.28 | 21.99 |
| 首组整步图 | 33 | 7 / 开启 | 30.87 / 32.49 / 33.75 | 32.37 |
| 后续独立复核 | 33 | 7 / 开启 | 23.35 / 26.10 / 30.83 | 26.76 |
| 轻量分项诊断 | 33 | 7 / 开启 | 24.31 / 22.76 / 29.71 / 30.94 / 31.10 | 27.77 |
| worker 保持活跃的诊断 | 33 | 7 / 开启 | 26.46 / 29.47 / 29.93 / 26.15 / 30.40 | 28.48 |
| 图内 DMA 路由诊断，最新 | 33 | 7 / 开启 | 27.21 / 23.94 / 27.65 / 29.56 / 29.67 | 27.61 |

这些数据来自不同开发阶段的构建，不能当作单一改动、同二进制的严格 A/B。
初始两组还使用了更小的缓存和不同 CPU 配置，不能将全部差值归因于内核优化。
最新代码尚未再做无插桩的最终验收；慢样本和未达标结果均保留。

33 专家配置均使用用户提供的初始 ID、7 个绑物理核的 worker、每 4 步自适应替换，
默认每轮全模型最多 8 次交换。首组整步图、独立复核与分段对照的各 384 个输出
token ID 完全一致。前三轮 assignment ratio 也一致，约为 40.52% / 48.33% / 52.07%。
最新五轮的 **640 个 token ID** 也与该对照输出一致。
在线学习状态跨轮保留，同一引擎中的各轮不视为独立进程样本。

这里统计真实 decode 调度步的 wall time，包含 CPU 专家、GPU 执行、传输、同步和在线替换。
每轮有 127 个 decode token；prefill 产生的首 token 不计入 decode 分子。
模型加载约两分钟，单独记录，不计入吞吐。

## 当前版本运行命令

```bash
cmake --build build --target nano-vllm-cpp-bench -j4
./build/nano-vllm-cpp-bench \
  --model build/perf-model-cache/Qwen3.5-35B-A3B-FP8 \
  --input-tokens 128 --output-tokens 128 --batch-sizes 1 \
  --warmup 1 --repeat 3 --num-blocks 32 --block-size 16 \
  --max-batched-tokens 128 --cpu-threads 7 \
  --gpu-expert-ids 0,2,4,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66 \
  --expert-adapt-every 4 --output decode30.json
```

该名单实际有 **33 个 ID**。全模型各层合计的专家权重与 scale 缓存约 3.87 GiB，
路由表、激活和递推状态另计。历史运行配置见 [原始记录](decode30-results.json)。

## 实现

- GPU 投影保留 checkpoint 的 FP8 权重及 FP32 scale，LM head 保留 BF16 权重。
  读取时恢复 FP32；激活、累加、KV 和递推状态仍是 FP32。
- AVX-512 用寄存器内查表得到精确 FP16 表示，再转换为 FP32。FP8 的全部有限值
  都可被该中间表示精确保存；缩放、FMA 链和最终归约顺序保持原实现。
- 单 token 的 GPU routed 专家用两次 kernel 提交完成 Gate/Up/SiLU 和 Down。
  设备路由表按实际专家 ID 查找当前槽位，因此缓存替换不要求重新捕获图。
- 单序列 greedy decode 捕获整步 GPU 计算。输入 ID、位置、长度、页表和状态槽位
  使用固定地址，每步更新内容。主线程从 mapped pinned 缓冲区读取路由和输入，
  批量运行 CPU 专家，再发布完成标志；GPU 按原 top-k 次序合并，并最后加入共享分支。
- 当前实验使用图内 DMA 下载路由和输入，再发布完成标志。CPU worker 在一整步
  decode 内保持就绪，步结束后仍可休眠。两项改动尚未得到稳定超过 30 tok/s 的结果。
- GPU 写回使用 system fence，CPU 读写完成标志使用 acquire/release。
  缓冲区在步末 stream 同步之后才复用。GPU 等待 CPU 的窗口为 200 ms；超时逐块
  记录错误，步末检查并抛出异常。CPU 异常会释放其余 GPU 等待节点后收拢执行。
- 图构建的 GPU 预热临时使用零 CPU 输出，随后恢复当前序列的卷积和递推缓存。
  活跃状态备份约 62.8 MiB；不会复制整个 KV 池。预热只写当前 fresh KV 位置，
  正式 replay 在同一位置重新写入，已缓存的前缀不变。
- 专家换入与 inference 使用同一 stream。CUDA 13.3 合并同层权重复制提交，旧版
  CUDA 保留逐参数复制；可变的 host 路由表 staging 由事件保护。步末等待实际上传完成。

整步图用于 Qwen3.5、单序列 greedy、top-k 不超过 8 且关闭专家 profiling 的 decode。
其他情况保留普通执行或分段图。设置 `NANOVLLM_DISABLE_WHOLE_GRAPH=1` 可关闭整步图，
`NANOVLLM_DISABLE_DECODE_GRAPHS=1` 可关闭 decode 图，用于对照。

## 检查与证据

- 紧凑 FP8/BF16 权重：120 组逐位对照，覆盖 decode、prefill、非对齐尺寸。
- 批量 GPU 专家：9 组与原逐专家执行逐位一致，包含真实 H=2048 / I=512 和尾部尺寸。
- CPU：覆盖全部 256 种 FP8 编码、归约、尾部、线程池批次和异常。
- MoE：零、部分、全部 GPU 专家，带/不带共享分支，stream 切换、在线换入和 usage 计数。
- 分段图和整步图各连续 32 步，对照输出、residual、卷积历史和递推状态，要求逐位一致。
- 阶段性的 CUDA memcheck、initcheck 均为 **0 errors**。这两次记录早于最后的
  worker 活跃期和 DMA 路由改动，不能声称覆盖当前全部改动。
- worker 活跃期的 CPU 并发/异常检查，以及 DMA 路由后的 MoE 原生一致性检查已通过。
  按用户要求停止后，未再运行 sanitizer 或性能测试。

```bash
cmake --build build --target nano-vllm-cpp-cpu-check \
  nano-vllm-cpp-compact-check nano-vllm-cpp-moe-sync-check -j4
./build/nano-vllm-cpp-cpu-check
./build/nano-vllm-cpp-compact-check
./build/nano-vllm-cpp-moe-sync-check
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/nano-vllm-cpp-moe-sync-check
compute-sanitizer --tool initcheck --error-exitcode 1 ./build/nano-vllm-cpp-moe-sync-check
```

原始结果位于 `profile/decode30-20261105/`：`before.json`、`compact3.json`、
`fast-poll33.json`、`whole33.json`、`final-confirm33.json`、`diagnostic33.json`、
`diagnostic-busy33.json`、`diagnostic-dma33.json`、`final-memcheck.txt`、`final-initcheck.txt`。
机器可读汇总见 [decode30-results.json](decode30-results.json)。
benchmark JSON 现同时保留每轮逐请求的 `token_ids`，便于复核输出。

分项诊断中未出现 major page fault。慢轮次的 CPU 专家关键路径约为 21–23 ms/token，
后续较快轮次约 12–14 ms/token；GPU 交接等待大致为 14–16 ms/token。
worker 保持活跃后，自愿上下文切换数量从约 1.5–1.9 万/轮降到约 0.5–0.7 万/轮，
但吞吐仍有波动。目前没有证据确认唯一根因。

这些吞吐结论限于上述本机、模型和负载；未验证 Windows 原生构建或其他长度的同等吞吐。
