cmake_minimum_required(VERSION 3.28)
project(nano-vllm-cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA translation units use C++20; CMake/NVCC do not support CUDA23 here.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# clangd cannot expand NVCC's --options-file include response files. Keep the
# same include directories directly in compile_commands.json instead.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

include(FetchContent)

find_package(Git REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(xxHash CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_program(CARGO_EXECUTABLE cargo HINTS "$ENV{HOME}/.cargo/bin")
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
  GIT_TAG v0.1.1
  GIT_SUBMODULES_RECURSE TRUE
  PATCH_COMMAND
  ${CMAKE_SOURCE_DIR}/scripts/apply_patch.sh
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/tokenizers-cpp-v0.1.1.patch
)
FetchContent_MakeAvailable(tokenizers_cpp)

# minja is header-only. Import its headers without its standalone build, which
# would fetch a second nlohmann/json and enable upstream examples/tests.
FetchContent_Declare(
  minja
  GIT_REPOSITORY https://github.com/google/minja.git
  GIT_TAG 021c2293c187789ef13d56c6cfd89c9b134fd80f
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
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--display-error-number --generate-line-info>
)

target_link_libraries(
  nano-vllm-cpp
  PRIVATE
  CUDA::cudart
  CUDA::cublas
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

set_target_properties(
  nano-vllm-cpp-cli
  PROPERTIES
  OUTPUT_NAME nano-vllm-cpp
)
