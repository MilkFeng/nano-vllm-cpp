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
