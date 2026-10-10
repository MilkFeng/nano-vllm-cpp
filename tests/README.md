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

测试检查缓存只分配指定数量的 GPU 专家，全部真实专家参数仍在 RAM，缓存
副本没有额外注册为模型参数。还检查填充完成前禁止使用缓存、GPU 副本独立
于 RAM 参数，以及显式重新填充后的结果。输出记录每次的专家编号、CPU/GPU
执行计数、槽位数量、GPU 权重字节数和最终输出误差。

```bash
./build/nano-vllm-cpp-moe-offload-test --cpu
./build/nano-vllm-cpp-moe-offload-test --gpu
compute-sanitizer --tool memcheck --error-exitcode 1 \
  ./build/nano-vllm-cpp-moe-offload-test --gpu
```
