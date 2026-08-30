#pragma once

#include <concepts>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "model/tensor.hh"

namespace nanovllm {

class Parameter {
 public:
  explicit Parameter(Tensor tensor);

  [[nodiscard]] Tensor& tensor() noexcept;
  [[nodiscard]] const Tensor& tensor() const noexcept;
  [[nodiscard]] TensorView view();
  [[nodiscard]] ConstTensorView view() const;

 private:
  Tensor tensor_;
};

class Module {
 public:
  using NamedParameter = std::pair<std::string, Parameter*>;
  using ConstNamedParameter = std::pair<std::string, const Parameter*>;
  using NamedParameters = std::vector<NamedParameter>;
  using ConstNamedParameters = std::vector<ConstNamedParameter>;

  Module() = default;
  virtual ~Module() = default;

  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&&) = delete;
  Module& operator=(Module&&) = delete;

  // Registries are non-owning. Registered objects must outlive this Module.
  void register_parameter(std::string name, Parameter& parameter);
  void register_module(std::string name, Module& module);

  // Returned pointers remain valid while the registered module tree lives.
  [[nodiscard]] NamedParameters named_parameters();
  [[nodiscard]] ConstNamedParameters named_parameters() const;

  [[nodiscard]] bool has_parameter(std::string_view full_name) const noexcept;
  [[nodiscard]] Parameter& parameter(std::string_view full_name);
  [[nodiscard]] const Parameter& parameter(std::string_view full_name) const;

 private:
  void append_named_parameters(std::string_view prefix,
                               NamedParameters& result);
  void append_named_parameters(std::string_view prefix,
                               ConstNamedParameters& result) const;
  [[nodiscard]] bool contains_module(const Module* target) const noexcept;

  std::map<std::string, Parameter*, std::less<>> parameters_;
  std::map<std::string, Module*, std::less<>> modules_;
};

class ModuleList : public Module {
 public:
  ModuleList() = default;

  template <std::derived_from<Module> T, typename... Args>
  T& emplace_back(Args&&... args) {
    auto child = std::make_unique<T>(std::forward<Args>(args)...);
    auto& result = *child;
    const auto name = std::to_string(modules_.size());

    modules_.push_back(std::move(child));
    try {
      register_module(name, result);
    } catch (...) {
      modules_.pop_back();
      throw;
    }
    return result;
  }

  [[nodiscard]] std::size_t size() const noexcept { return modules_.size(); }

  [[nodiscard]] Module& operator[](std::size_t index) noexcept {
    return *modules_[index];
  }

  [[nodiscard]] const Module& operator[](std::size_t index) const noexcept {
    return *modules_[index];
  }

  [[nodiscard]] Module& at(std::size_t index) { return *modules_.at(index); }

  [[nodiscard]] const Module& at(std::size_t index) const {
    return *modules_.at(index);
  }

 private:
  std::vector<std::unique_ptr<Module>> modules_;
};

} // namespace nanovllm
