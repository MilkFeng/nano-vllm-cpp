#include <cuda_runtime_api.h>

#include "common/cuda_utils.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

GPUInfo gpu_info() {
  const CudaDeviceGuard guard(0);
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0), "read GPU properties");
  GPUInfo info{.name = properties.name,
               .total_memory_bytes = properties.totalGlobalMem,
               .compute_capability_major = properties.major,
               .compute_capability_minor = properties.minor};
  check_cuda(cudaRuntimeGetVersion(&info.cuda_runtime_version),
             "read CUDA runtime version");
  check_cuda(cudaDriverGetVersion(&info.cuda_driver_version),
             "read CUDA driver version");
  return info;
}

GPUMemoryInfo gpu_memory_info() {
  const CudaDeviceGuard guard(0);
  GPUMemoryInfo info;
  check_cuda(cudaMemGetInfo(&info.free_bytes, &info.total_bytes),
             "read GPU memory");
  return info;
}

} // namespace nanovllm
