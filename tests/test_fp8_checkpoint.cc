#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "core/checkpoint.hh"
#include "core/model.hh"
#include "kernels/cpu/fp8.hh"

namespace {

using namespace nanovllm;
using nlohmann::json;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void expect_throw(Function&& function) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error("expected exception was not thrown");
}

struct TestDirectory {
  std::filesystem::path path;

  TestDirectory() {
    auto pattern =
        (std::filesystem::temp_directory_path() / "nanovllm-fp8-XXXXXX")
            .string();
    const auto* directory = ::mkdtemp(pattern.data());
    if (directory == nullptr)
      throw std::runtime_error("failed to create test directory");
    path = directory;
  }
  ~TestDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

json tensor_header(std::string_view dtype, std::vector<std::size_t> shape,
                   std::size_t begin, std::size_t end) {
  return {{"dtype", dtype},
          {"shape", std::move(shape)},
          {"data_offsets", {begin, end}}};
}

void write_safetensors(const std::filesystem::path& path, const json& header,
                       std::span<const std::byte> data) {
  auto text = header.dump();
  text.append((8 - text.size() % 8) % 8, ' ');
  std::ofstream file(path, std::ios::binary);
  const auto length = static_cast<std::uint64_t>(text.size());
  for (unsigned int i = 0; i < 8; ++i)
    file.put(static_cast<char>((length >> (i * 8)) & 0xff));
  file.write(text.data(), static_cast<std::streamsize>(text.size()));
  file.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size()));
  require(static_cast<bool>(file), "failed to write safetensors fixture");
}

void write_index(const std::filesystem::path& path, const json& weights) {
  std::ofstream file(path);
  file << json{{"metadata", {{"total_size", 0}}}, {"weight_map", weights}};
  require(static_cast<bool>(file), "failed to write checkpoint index");
}

constexpr std::size_t kRows = 129, kColumns = 257;
constexpr std::array<std::uint8_t, 4> kCodes{0x38, 0xb8, 0x40, 0x01};
constexpr std::array<float, 4> kValues{1.0f, -1.0f, 2.0f, 1.0f / 512};
constexpr std::array<float, 6> kScales{0.5f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f};

struct Fixture {
  TestDirectory directory;
  std::vector<std::uint8_t> weights =
      std::vector<std::uint8_t>(kRows * kColumns);
  json index;

  Fixture() {
    for (std::size_t i = 0; i < weights.size(); ++i)
      weights[i] = kCodes[i % kCodes.size()];
    // Different dtypes share each shard; scales are in a different file from
    // weights.
    const std::array<std::uint16_t, 2> bf16{0x3f80, 0xc000};
    std::vector<std::byte> first(std::as_bytes(std::span(bf16)).begin(),
                                 std::as_bytes(std::span(bf16)).end());
    const auto weight_bytes = std::as_bytes(std::span(weights));
    first.insert(first.end(), weight_bytes.begin(), weight_bytes.end());
    write_safetensors(
        directory.path / "weights.safetensors",
        {{"router.weight", tensor_header("BF16", {1, 2}, 0, 4)},
         {"expert.weight",
          tensor_header("F8_E4M3", {kRows, kColumns}, 4, first.size())}},
        first);

    std::vector<std::byte> second(std::as_bytes(std::span(kScales)).begin(),
                                  std::as_bytes(std::span(kScales)).end());
    const std::array<std::int32_t, 1> integer{7};
    const auto integer_bytes = std::as_bytes(std::span(integer));
    second.insert(second.end(), integer_bytes.begin(), integer_bytes.end());
    write_safetensors(
        directory.path / "scales.safetensors",
        {{"expert.weight_scale_inv", tensor_header("F32", {2, 3}, 0, 24)},
         {"integer", tensor_header("I32", {1}, 24, 28)}},
        second);
    index = {{"expert.weight", "weights.safetensors"},
             {"expert.weight_scale_inv", "scales.safetensors"},
             {"router.weight", "weights.safetensors"},
             {"integer", "scales.safetensors"}};
    write_index(directory.path / "model.safetensors.index.json", index);
  }
};

class PackedModel final : public Model {
 public:
  PackedModel()
      : weight_(Tensor::empty({kRows, kColumns}, DataType::Float8E4M3FN)),
        scale_(Tensor::empty({2, 3}, DataType::Float32)) {
    expert_.register_parameter("weight", weight_);
    expert_.register_parameter("weight_scale_inv", scale_);
    register_module("expert", expert_);
    std::memset(weight_.view().data(), 0x55, weight_.view().size_bytes());
    std::fill_n(scale_.view().data<float>(), 6, 42.0f);
  }

 private:
  Parameter weight_;
  Parameter scale_;
  Module expert_;
};

void test_decode() {
  const std::array<std::pair<std::uint8_t, float>, 14> cases{{
      {0x00, 0.0f},
      {0x80, -0.0f},
      {0x01, 1.0f / 512},
      {0x07, 7.0f / 512},
      {0x08, 1.0f / 64},
      {0x30, 0.5f},
      {0x38, 1.0f},
      {0x3c, 1.5f},
      {0x40, 2.0f},
      {0x78, 256.0f},
      {0x7e, 448.0f},
      {0xb8, -1.0f},
      {0x81, -1.0f / 512},
      {0xfe, -448.0f},
  }};
  for (const auto& [bits, expected] : cases) {
    const auto actual = kernels::cpu::fp8_e4m3fn_to_float(bits);
    require(
        actual == expected && std::signbit(actual) == std::signbit(expected),
        "FP8 edge value mismatch");
  }
  require(std::isnan(kernels::cpu::fp8_e4m3fn_to_float(0x7f)),
          "positive FP8 NaN mismatch");
  require(std::isnan(kernels::cpu::fp8_e4m3fn_to_float(0xff)),
          "negative FP8 NaN mismatch");
  require(data_type_size(DataType::Float8E4M3FN) == 1,
          "FP8 storage must use one byte");
  require(json(DataType::Float8E4M3FN) == "F8_E4M3",
          "FP8 safetensors spelling mismatch");
  require(json("F8_E4M3").get<DataType>() == DataType::Float8E4M3FN,
          "FP8 dtype parsing failed");
  expect_throw<std::invalid_argument>(
      [] { (void)json("F8_E4M3FNUZ").get<DataType>(); });
}

void test_checkpoint_and_dequantize() {
  Fixture fixture;
  Checkpoint checkpoint(fixture.directory.path);
  require(checkpoint.weight_map().size() == 4, "index tensor count mismatch");
  const auto& weight_info = checkpoint.tensor("expert.weight");
  const auto bytes = checkpoint.raw_data("expert.weight");
  require(weight_info.dtype == DataType::Float8E4M3FN
              && weight_info.shape == std::vector<std::size_t>{kRows, kColumns},
          "FP8 metadata mismatch");
  require(bytes.size() == fixture.weights.size()
              && std::memcmp(bytes.data(), fixture.weights.data(), bytes.size())
                     == 0,
          "mmap weight bytes mismatch");

  auto scales = Tensor::empty({2, 3}, DataType::Float32);
  const auto scale_bytes = checkpoint.raw_data("expert.weight_scale_inv");
  std::memcpy(scales.data(), scale_bytes.data(), scales.size_bytes());
  require(checkpoint.tensor("router.weight").dtype == DataType::BFloat16
              && checkpoint.tensor("integer").dtype == DataType::Int32,
          "mixed checkpoint dtypes mismatch");
  require(checkpoint.raw_data("expert.weight").data() == bytes.data(),
          "opening another shard invalidated a span");
  require(&checkpoint.tensor("expert.weight") == &weight_info,
          "metadata reference invalidated");
  expect_throw<std::out_of_range>([&] { (void)checkpoint.tensor("unknown"); });

  ConstTensorView weight(bytes.data(), weight_info.dtype,
                         TensorLayout(weight_info.shape), Device::cpu());
  auto output = Tensor::empty({kRows, kColumns}, DataType::Float32);
  kernels::cpu::dequantize_fp8(weight, scales.view().as_const(), output.view());
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t column = 0; column < kColumns; ++column) {
      const auto index = row * kColumns + column;
      const auto expected = kValues[index % kValues.size()]
                            * kScales[(row / 128) * 3 + column / 128];
      require(output.view().data<float>()[index] == expected,
              "128x128 scale or tail block mismatch");
    }
  }
  // Reuse the output allocation with a changed scale matrix.
  scales.view().data<float>()[5] = 3.0f;
  const auto* output_pointer = output.data();
  kernels::cpu::dequantize_fp8(weight, scales.view().as_const(), output.view());
  require(output.data() == output_pointer
              && output.view().data<float>()[128 * kColumns + 256]
                     == kValues[(128 * kColumns + 256) % 4] * 3.0f,
          "dequantization workspace reuse failed");

  Checkpoint direct_index(fixture.directory.path
                          / "model.safetensors.index.json");
  require(direct_index.raw_data("expert.weight").size() == bytes.size(),
          "direct index path failed");
  Checkpoint single(fixture.directory.path / "weights.safetensors");
  require(single.weight_map().size() == 2
              && single.raw_data("expert.weight").size() == bytes.size(),
          "single safetensors file failed");

  PackedModel model;
  expect_throw<std::runtime_error>(
      [&] { (void)model.load_ckpt(fixture.directory.path); });
  require(
      model.parameter("expert.weight").view().data<std::uint8_t>()[0] == 0x55
          && model.parameter("expert.weight_scale_inv").view().data<float>()[0]
                 == 42.0f,
      "strict validation copied before rejecting unexpected tensors");
  const auto result = model.load_ckpt(fixture.directory.path,
                                      {.allow_unexpected_tensors = true});
  require(result.num_tensors == 2 && result.num_bytes == bytes.size() + 24,
          "sharded Model load statistics mismatch");
  require(std::memcmp(model.parameter("expert.weight").view().data(),
                      fixture.weights.data(), bytes.size())
              == 0,
          "sharded Model FP8 copy mismatch");
  require(std::memcmp(model.parameter("expert.weight_scale_inv").view().data(),
                      kScales.data(), 24)
              == 0,
          "sharded Model scale copy mismatch");
}

void test_lazy_shards() {
  Fixture fixture;
  fixture.index["late"] = "late.safetensors";
  fixture.index["wrong"] = "weights.safetensors";
  write_index(fixture.directory.path / "model.safetensors.index.json",
              fixture.index);
  Checkpoint checkpoint(fixture.directory.path);
  require(checkpoint.raw_data("expert.weight").size() == fixture.weights.size(),
          "unused missing shard prevented available tensor access");
  expect_throw<std::system_error>([&] { (void)checkpoint.tensor("late"); });
  expect_throw<std::out_of_range>([&] { (void)checkpoint.tensor("wrong"); });
  const std::array<float, 1> value{2.0f};
  write_safetensors(fixture.directory.path / "late.safetensors",
                    {{"late", tensor_header("F32", {1}, 0, 4)}},
                    std::as_bytes(std::span(value)));
  require(checkpoint.tensor("late").dtype == DataType::Float32,
          "failed shard access could not be retried after file arrived");
}

void test_invalid_inputs() {
  auto weight = Tensor::empty({3, 5}, DataType::Float8E4M3FN);
  std::fill_n(weight.view().data<std::uint8_t>(), 15, 0x38);
  auto scales = Tensor::empty({2, 2}, DataType::Float32);
  std::copy_n(kScales.data(), 4, scales.view().data<float>());
  auto output = Tensor::empty({3, 5}, DataType::Float32);
  const auto run = [&](ConstTensorView w, ConstTensorView s, TensorView y,
                       std::size_t br, std::size_t bc) {
    kernels::cpu::dequantize_fp8(w, s, y, br, bc);
  };
  run(weight.view().as_const(), scales.view().as_const(), output.view(), 2, 3);
  require(output.view().data<float>()[0] == 0.5f
              && output.view().data<float>()[3] == 2.0f
              && output.view().data<float>()[10] == 4.0f
              && output.view().data<float>()[14] == 8.0f,
          "custom FP8 blocks failed");
  expect_throw<std::invalid_argument>([&] {
    run(weight.view().as_const(), scales.view().as_const(), output.view(), 0,
        3);
  });
  expect_throw<std::invalid_argument>([&] {
    run(weight.view().as_const(), scales.view().as_const(), output.view(), 2,
        0);
  });
  expect_throw<std::invalid_argument>([&] {
    run(weight.view().as_const(), scales.view().as_const(), output.view(), 128,
        128);
  });
  expect_throw<std::invalid_argument>([&] {
    run(output.view().as_const(), scales.view().as_const(), output.view(), 2,
        3);
  });
  expect_throw<std::invalid_argument>([&] {
    run(weight.view().as_const(), weight.view().as_const(), output.view(), 2,
        3);
  });
  expect_throw<std::invalid_argument>([&] {
    run(weight.view().as_const(), scales.view().as_const(), weight.view(), 2,
        3);
  });
  auto wrong = Tensor::empty({3, 4}, DataType::Float32);
  expect_throw<std::invalid_argument>([&] {
    run(weight.view().as_const(), scales.view().as_const(), wrong.view(), 2, 3);
  });
  ConstTensorView strided(weight.data(), weight.dtype(),
                          TensorLayout({3, 5}, {6, 1}), Device::cpu());
  expect_throw<std::invalid_argument>(
      [&] { run(strided, scales.view().as_const(), output.view(), 2, 3); });
  auto empty = Tensor::empty({0, 5}, DataType::Float8E4M3FN);
  expect_throw<std::invalid_argument>([&] {
    run(empty.view().as_const(), scales.view().as_const(), output.view(), 2, 3);
  });
  auto one_scale = Tensor::empty({1, 1}, DataType::Float32);
  one_scale.view().data<float>()[0] = 2.0f;
  run(weight.view().as_const(), one_scale.view().as_const(), output.view(),
      std::numeric_limits<std::size_t>::max(),
      std::numeric_limits<std::size_t>::max());
  require(output.view().data<float>()[14] == 2.0f,
          "FP8 ceil division overflowed");

  TestDirectory directory;
  const std::array<std::byte, 1> byte{std::byte{0x38}};
  const auto path = directory.path / "bad.safetensors";
  write_safetensors(path, {{"weight", tensor_header("F8_E4M3", {2}, 0, 1)}},
                    byte);
  expect_throw<std::runtime_error>([&] { Checkpoint bad(path); });
  write_safetensors(path, {{"weight", tensor_header("F8_E5M2", {1}, 0, 1)}},
                    byte);
  expect_throw<std::invalid_argument>([&] { Checkpoint bad(path); });
  write_safetensors(path, {{"weight", tensor_header("F8_E4M3", {1}, 1, 2)}},
                    byte);
  expect_throw<std::runtime_error>([&] { Checkpoint bad(path); });
  write_safetensors(path,
                    {{"a", tensor_header("F8_E4M3", {1}, 0, 1)},
                     {"b", tensor_header("F8_E4M3", {1}, 0, 1)}},
                    byte);
  expect_throw<std::runtime_error>([&] { Checkpoint bad(path); });
  write_safetensors(directory.path / "model.safetensors",
                    {{"weight", tensor_header("F8_E4M3", {1}, 0, 1)}}, byte);
  Checkpoint unsharded_directory(directory.path);
  require(unsharded_directory.tensor("weight").byte_size() == 1,
          "single-file directory fallback failed");
  const auto index_path = directory.path / "model.safetensors.index.json";
  write_index(index_path, json::object());
  expect_throw<std::runtime_error>([&] { Checkpoint bad(directory.path); });
  write_index(index_path, {{"weight", ""}});
  expect_throw<std::runtime_error>([&] { Checkpoint bad(index_path); });
  write_index(index_path, {{"weight", "../bad.safetensors"}});
  expect_throw<std::runtime_error>([&] { Checkpoint bad(index_path); });
}

} // namespace

int main() {
  try {
    test_decode();
    test_checkpoint_and_dequantize();
    test_lazy_shards();
    test_invalid_inputs();
    std::cout << "FP8 checkpoint CPU checks passed: decode, sharded mmap, "
                 "block scales, Model loading, lazy retries and validation\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
