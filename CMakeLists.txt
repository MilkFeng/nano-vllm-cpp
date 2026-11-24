cmake_minimum_required(VERSION 3.28)
project(nano-vllm-cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

include(FetchContent)

if(MSVC)
  set(TOKENIZERS_CPP_MSVC_RUNTIME_LIBRARY "MD" CACHE STRING "")
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
  )
endif()

find_package(Git REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(xxHash CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_program(CARGO_EXECUTABLE cargo HINTS
  "$ENV{CARGO_HOME}/bin" "$ENV{USERPROFILE}/.cargo/bin" "$ENV{HOME}/.cargo/bin")
if(NOT CARGO_EXECUTABLE)
  message(
    FATAL_ERROR
    "tokenizers-cpp requires Rust and Cargo; install them before configuring"
  )
endif()

set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  tokenizers_cpp
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
  GIT_TAG main
  GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(tokenizers_cpp)

FetchContent_Declare(
  minja
  GIT_REPOSITORY https://github.com/google/minja.git
  GIT_TAG main
  SOURCE_SUBDIR include
)
FetchContent_MakeAvailable(minja)

file(
  GLOB_RECURSE NANO_VLLM_CPP_SOURCES
  CONFIGURE_DEPENDS

  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# ========== nano-vllm-cpp library ==========

add_library(nano-vllm-cpp STATIC)
add_library(nano-vllm-cpp::nano-vllm-cpp ALIAS nano-vllm-cpp)

target_sources(
  nano-vllm-cpp
  PRIVATE
  ${NANO_VLLM_CPP_SOURCES}
)

target_include_directories(
  nano-vllm-cpp
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(nano-vllm-cpp SYSTEM PRIVATE ${minja_SOURCE_DIR}/include)

target_compile_options(
  nano-vllm-cpp
  PRIVATE
  "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4>"
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--display-error-number;--generate-line-info>"
  PUBLIC
  "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/arch:AVX2>"
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-mavx2;-mfma>"
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/arch:AVX2>"
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Xcompiler=-mavx2,-mfma>"
)

# Keep AVX-512 isolated: the dispatcher and the rest of the binary use AVX2.
set_source_files_properties(src/kernels/cpu/fp8_matvec_avx512.cc PROPERTIES
  COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX512>;$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl>")

if(WIN32)
  target_compile_definitions(nano-vllm-cpp PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_link_libraries(
  nano-vllm-cpp
  PRIVATE
  Threads::Threads
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  nlohmann_json::nlohmann_json
  spdlog::spdlog
  tokenizers_cpp
  xxHash::xxhash
  magic_enum::magic_enum
)

# ========== nano-vllm-cpp CLI ==========

add_executable(
  nano-vllm-cpp-cli
  app/main.cc
)

target_link_libraries(
  nano-vllm-cpp-cli
  PRIVATE
  nano-vllm-cpp::nano-vllm-cpp
  CLI11::CLI11
  nlohmann_json::nlohmann_json
)


add_executable(nano-vllm-cpp-bench app/benchmark.cc)

target_compile_definitions(nano-vllm-cpp-bench PRIVATE
  NANOVLLM_BENCH_BUILD_TYPE="$<CONFIG>")

target_link_libraries(nano-vllm-cpp-bench PRIVATE
  nano-vllm-cpp::nano-vllm-cpp CLI11::CLI11 nlohmann_json::nlohmann_json)

# =========== COMP KERNEL ===========

add_executable(nano-vllm-cpp-comp-gemm EXCLUDE_FROM_ALL
  exp/gemm/comp_gemm.cc
  exp/gemm/comp_gemm_main.cc
  exp/gemm/gemm_benchmark.cc
  exp/gemm/gemm_baseline.cu
)

target_include_directories(nano-vllm-cpp-comp-gemm PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(nano-vllm-cpp-comp-gemm PRIVATE
  nano-vllm-cpp::nano-vllm-cpp CUDA::cudart CUDA::cublas CUDA::cublasLt
  CLI11::CLI11 nlohmann_json::nlohmann_json
)

target_compile_definitions(nano-vllm-cpp-comp-gemm PRIVATE
  NANOVLLM_GEMM_BUILD_TYPE="$<CONFIG>"
  NANOVLLM_GEMM_CUDA_COMPILER="${CMAKE_CUDA_COMPILER_VERSION}"
)

add_library(nano-vllm-cpp-attn-exp SHARED EXCLUDE_FROM_ALL
  exp/attn/attn_bridge.cc
  src/kernels/cuda/attn/attn.cu
  src/core/device.cc
  src/core/tensor_layout.cc
)

target_include_directories(nano-vllm-cpp-attn-exp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(nano-vllm-cpp-attn-exp PRIVATE CUDA::cudart)
target_compile_options(nano-vllm-cpp-attn-exp PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>"
)
target_compile_definitions(nano-vllm-cpp-attn-exp PRIVATE
  NANOVLLM_ATTN_BUILD_TYPE="$<CONFIG>"
)

add_executable(nano-vllm-cpp-attn-check EXCLUDE_FROM_ALL exp/attn/check_native.cc)
target_include_directories(nano-vllm-cpp-attn-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nano-vllm-cpp-attn-check PRIVATE nano-vllm-cpp-attn-exp CUDA::cudart)

add_executable(nano-vllm-cpp-wmma-check EXCLUDE_FROM_ALL exp/attn/check_wmma.cu)
target_include_directories(nano-vllm-cpp-wmma-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nano-vllm-cpp-wmma-check PRIVATE CUDA::cudart)

# CPU checks/benchmarks do not create a CUDA context.
add_executable(nano-vllm-cpp-cpu-check EXCLUDE_FROM_ALL exp/cpu/check.cc)
target_include_directories(nano-vllm-cpp-cpu-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nano-vllm-cpp-cpu-check PRIVATE nano-vllm-cpp::nano-vllm-cpp)
add_executable(nano-vllm-cpp-cpu-bench EXCLUDE_FROM_ALL exp/cpu/benchmark.cc)
target_include_directories(nano-vllm-cpp-cpu-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nano-vllm-cpp-cpu-bench PRIVATE
  nano-vllm-cpp::nano-vllm-cpp nlohmann_json::nlohmann_json)

add_executable(nano-vllm-cpp-moe-sync-check EXCLUDE_FROM_ALL exp/moe/check.cc)
target_include_directories(nano-vllm-cpp-moe-sync-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nano-vllm-cpp-moe-sync-check PRIVATE
  nano-vllm-cpp::nano-vllm-cpp CUDA::cudart nlohmann_json::nlohmann_json)

add_executable(nano-vllm-cpp-compact-check EXCLUDE_FROM_ALL exp/gemm/check_compact.cc)
target_include_directories(nano-vllm-cpp-compact-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nano-vllm-cpp-compact-check PRIVATE nano-vllm-cpp::nano-vllm-cpp CUDA::cudart)

add_executable(nano-vllm-cpp-precision-check EXCLUDE_FROM_ALL exp/precision/check.cc)
target_include_directories(nano-vllm-cpp-precision-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nano-vllm-cpp-precision-check PRIVATE
  nano-vllm-cpp::nano-vllm-cpp CUDA::cudart nlohmann_json::nlohmann_json)
