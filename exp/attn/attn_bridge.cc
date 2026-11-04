#include "attn_bridge.hh"

#include <stdexcept>
#include <string>
#include <vector>

#include "core/tensor_view.hh"
#include "kernels/cuda/kernels.hh"

namespace {

thread_local std::string last_error;

nanovllm::DataType dtype_for(int value) {
  switch (value) {
    case 0:
      return nanovllm::DataType::Float32;
    case 1:
      return nanovllm::DataType::Float16;
    case 2:
      return nanovllm::DataType::BFloat16;
    default:
      throw std::invalid_argument("unknown attention dtype");
  }
}

void validate_dimensions(const NanoAttnArgs& args) {
  if (args.batch < 0 || args.tokens < 0 || args.query_heads <= 0
      || args.kv_heads <= 0 || args.query_heads % args.kv_heads != 0
      || args.head_dim <= 0 || args.head_dim > 256 || args.pages <= 0
      || args.page_size <= 0 || args.table_width < 0 || args.max_query < 0
      || args.max_key < 0) {
    throw std::invalid_argument("invalid attention dimensions");
  }
}

void forward(const NanoAttnArgs& args) {
  using namespace nanovllm;
  validate_dimensions(args);
  if (args.workspace == nullptr && args.workspace_elements != 0) {
    throw std::invalid_argument("null attention workspace");
  }

  const auto dtype = dtype_for(args.dtype);
  const auto view = [&](void* pointer, DataType type,
                        std::vector<std::size_t> shape) {
    return TensorView(pointer, type, TensorLayout(std::move(shape)),
                      Device::cuda(0));
  };
  const auto qshape =
      std::vector<std::size_t>{static_cast<std::size_t>(args.tokens),
                               static_cast<std::size_t>(args.query_heads),
                               static_cast<std::size_t>(args.head_dim)};
  const auto new_shape =
      std::vector<std::size_t>{static_cast<std::size_t>(args.tokens),
                               static_cast<std::size_t>(args.kv_heads),
                               static_cast<std::size_t>(args.head_dim)};
  const auto cache_shape =
      std::vector<std::size_t>{static_cast<std::size_t>(args.pages),
                               static_cast<std::size_t>(args.page_size),
                               static_cast<std::size_t>(args.kv_heads),
                               static_cast<std::size_t>(args.head_dim)};

  auto query = view(args.query, dtype, qshape);
  auto output = view(args.output, dtype, qshape);
  auto key_cache = view(args.key_cache, dtype, cache_shape);
  auto value_cache = view(args.value_cache, dtype, cache_shape);
  auto key = view(args.new_key, dtype, new_shape);
  auto value = view(args.new_value, dtype, new_shape);
  auto cu_query = view(args.cu_query, DataType::Int32,
                       {static_cast<std::size_t>(args.batch) + 1});
  auto cu_key = view(args.cu_key, DataType::Int32,
                     {static_cast<std::size_t>(args.batch) + 1});
  auto table = view(args.block_table, DataType::Int32,
                    {static_cast<std::size_t>(args.batch),
                     static_cast<std::size_t>(args.table_width)});
  auto workspace = view(args.workspace, DataType::Float32,
                        {static_cast<std::size_t>(args.workspace_elements)});

  kernels::cuda::flash_attn_with_kvcache(
      query.as_const(), key_cache, value_cache, key.as_const(),
      value.as_const(), cu_query.as_const(), cu_key.as_const(), args.max_query,
      args.max_key, table.as_const(), output, args.causal != 0,
      static_cast<cudaStream_t>(args.stream), workspace);
}

} // namespace

int nano_attn_workspace_size(const NanoAttnArgs* args,
                             std::uint64_t* elements) noexcept {
  try {
    if (args == nullptr || elements == nullptr) {
      throw std::invalid_argument("null attention workspace size arguments");
    }
    validate_dimensions(*args);
    *elements =
        args->tokens > 0
            ? nanovllm::kernels::cuda::flash_attn_workspace_elements(
                  args->batch, args->query_heads, args->head_dim, args->max_key)
            : 0;
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}

int nano_attn_forward(const NanoAttnArgs* args) noexcept {
  try {
    if (args == nullptr) {
      throw std::invalid_argument("null attention arguments");
    }

    forward(*args);
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}

const char* nano_attn_error() noexcept {
  return last_error.c_str();
}

const char* nano_attn_build_type() noexcept {
  return NANOVLLM_ATTN_BUILD_TYPE;
}
