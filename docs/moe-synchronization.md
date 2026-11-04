# MoE 同步优化

本次调整针对 B=1 decode：合并路由 ID 和输入下载，把共享专家提前到 CPU
专家计算期间执行，并允许普通 MoE forward 在 GPU 工作完成前返回。
专家选择、路由权重、FP8/FP32 运算和 combine 顺序保持原有实现。

## 执行顺序

原有混合执行层依次等待路由 ID 下载、输入下载和层末 GPU 工作完成。
当前路径在路由后连续提交 IDs 与输入的 D2H，记录一个 host-ready event，
随后立即提交共享专家，再等待这个 event。CPU 拿到 IDs 和输入后提交专家
批次，同时在 GPU 上排入命中专家；CPU 批次完成后仍按层一次上传输出，
接着提交 scatter、combine 和共享分支相加。

H=2048 的 FP32 单 token 输入为 8 KiB。部分专家常驻时，即使本次 top-k
全部命中，也会预先下载这 8 KiB；所有专家都常驻时跳过输入下载。
多 token prefill 保留按实际 miss 下载输入的批量 executor 和共享分支顺序，
同样移除普通 forward 的层末 stream 同步。

## 生命周期和计时

`MoEExecutionWorkspace` 复用两个禁用计时的 CUDA event。host-ready event
之前的 stream 工作完成后，CPU 才能读取 IDs/输入，并覆盖旧的 host_order、
host_parts。层末 completion event 保护仍在被 H2D 读取的 pinned staging，
以及仍被 kernel 使用的 GPU 工作区。

同一 stream 的下一次 MoE 下载事件已经覆盖前一次的工作，因此不增加层间
等待。切换 stream 或独立调用 `ExpertExecutor::forward_into()` 时，先等待
旧的工作区使用完成；工作区析构也等待最后一次完成事件。调用必须串行，
不支持多个主线程同时使用同一份工作区。

返回的 Tensor 继续拥有独立存储。调用方须保持输入、模块和输出在对应 GPU
操作完成前有效；引擎的临时 Tensor 池按同一 stream 顺序复用存储，并在每个
推理步下载采样结果后同步。异常路径仍先收拢 CPU 任务并同步 stream。
在线专家替换的上传与专家 kernel 使用同一 inference stream，更新槽位映射前
等待上传完成。

`--profile-experts` 在层末等待 completion event，继续记录包含 GPU 完成等待的
MoE wall time。正式性能对照关闭 profiling；profiling 时间不作为无等待路径
的吞吐结果。

## 检查

```bash
cmake --build build --target nano-vllm-cpp-moe-sync-check nano-vllm-cpp-bench -j 4
./build/nano-vllm-cpp-moe-sync-check
compute-sanitizer --tool memcheck ./build/nano-vllm-cpp-moe-sync-check
```

原生检查对照 CPU 专家和独立标量路由/共享分支，覆盖零、部分、全部常驻专家，
共享分支开关、单 token 与多 token 交替、工作区扩容、两层共用工作区、切换
非默认 stream、profiling/在线使用计数和独立 executor 工作区析构。

整模型对照固定 33 个 GPU 专家/层、8 个 CPU worker、每 4 步自适应替换、
输入/输出各 128 token、B=1、相同 prompt 与 seed，预热 1 轮并测量 3 轮。
修改前静态库和源码冻结后，与修改版使用相同计数插桩，记录同步次数、CPU
任务等待、吞吐和全部输出 token。原始记录位于 `profile/moe-sync-20261104/`。
