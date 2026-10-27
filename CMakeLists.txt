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
