#include "core/module.hh"

#include <cassert>
#include <stdexcept>

namespace nanovllm {
namespace {

std::string join_name(std::string_view prefix, std::string_view local_name) {
  if (prefix.empty()) {
    return std::string(local_name);
  }

  std::string result;
  result.reserve(prefix.size() + 1 + local_name.size());
  result.append(prefix);
  result.push_back('.');
  result.append(local_name);
  return result;
}

} // namespace

Parameter::Parameter(Tensor tensor)
    : tensor_(std::move(tensor)) {}

Tensor& Parameter::tensor() noexcept {
  return tensor_;
}

const Tensor& Parameter::tensor() const noexcept {
  return tensor_;
}

TensorView Parameter::view() {
  return tensor_.view();
}

ConstTensorView Parameter::view() const {
  return tensor_.view();
}

void Module::register_parameter(std::string name, Parameter& parameter) {
  assert(!name.empty() && name.find('.') == std::string_view::npos);

  if (parameters_.contains(name) || modules_.contains(name)) {
    throw std::invalid_argument("module name was registered twice: " + name);
  }

  parameters_.emplace(std::move(name), &parameter);
}

void Module::register_module(std::string name, Module& module) {
  assert(!name.empty() && name.find('.') == std::string_view::npos);
  assert(module != this && !module.contains_module(this));

  if (parameters_.contains(name) || modules_.contains(name)) {
    throw std::invalid_argument("module name was registered twice: " + name);
  }

  modules_.emplace(std::move(name), &module);
}

Module::NamedParameters Module::named_parameters() {
  NamedParameters result;
  append_named_parameters({}, result);
  return result;
}

Module::ConstNamedParameters Module::named_parameters() const {
  ConstNamedParameters result;
  append_named_parameters({}, result);
  return result;
}

bool Module::has_parameter(std::string_view full_name) const noexcept {
  if (full_name.empty()) {
    return false;
  }

  const auto separator = full_name.find('.');
  if (separator == std::string_view::npos) {
    return parameters_.contains(full_name);
  }

  const auto local_name = full_name.substr(0, separator);
  const auto remaining_name = full_name.substr(separator + 1);
  if (local_name.empty() || remaining_name.empty()) {
    return false;
  }

  const auto child = modules_.find(local_name);
  return child != modules_.end()
         && child->second->has_parameter(remaining_name);
}

Parameter& Module::parameter(std::string_view full_name) {
  return const_cast<Parameter&>(
      static_cast<const Module&>(*this).parameter(full_name));
}

const Parameter& Module::parameter(std::string_view full_name) const {
  const auto separator = full_name.find('.');
  if (separator == std::string_view::npos) {
    const auto found = parameters_.find(full_name);
    if (found != parameters_.end()) {
      return *found->second;
    }
  } else {
    const auto local_name = full_name.substr(0, separator);
    const auto remaining_name = full_name.substr(separator + 1);
    if (!local_name.empty() && !remaining_name.empty()) {
      const auto child = modules_.find(local_name);
      if (child != modules_.end()) {
        return child->second->parameter(remaining_name);
      }
    }
  }

  throw std::out_of_range("module parameter was not registered: "
                          + std::string(full_name));
}

Module& Module::module(std::string_view full_name) {
  return const_cast<Module&>(std::as_const(*this).module(full_name));
}

const Module& Module::module(std::string_view full_name) const {
  if (full_name.empty()) return *this;
  const auto separator = full_name.find('.');
  const auto found = modules_.find(full_name.substr(0, separator));
  if (found == modules_.end()) {
    throw std::out_of_range("module was not registered: " + std::string(full_name));
  }
  return separator == std::string_view::npos
             ? *found->second
             : std::as_const(*found->second).module(full_name.substr(separator + 1));
}

void Module::append_named_parameters(std::string_view prefix,
                                     NamedParameters& result) {
  for (const auto& [name, parameter] : parameters_) {
    result.emplace_back(join_name(prefix, name), parameter);
  }

  for (const auto& [name, module] : modules_) {
    const auto child_prefix = join_name(prefix, name);
    module->append_named_parameters(child_prefix, result);
  }
}

void Module::append_named_parameters(std::string_view prefix,
                                     ConstNamedParameters& result) const {
  for (const auto& [name, parameter] : parameters_) {
    result.emplace_back(join_name(prefix, name), parameter);
  }

  for (const auto& [name, module] : modules_) {
    const auto child_prefix = join_name(prefix, name);
    static_cast<const Module&>(*module).append_named_parameters(child_prefix,
                                                                result);
  }
}

bool Module::contains_module(const Module* target) const noexcept {
  if (this == target) {
    return true;
  }

  for (const auto& [name, module] : modules_) {
    if (module->contains_module(target)) {
      return true;
    }
  }
  return false;
}

} // namespace nanovllm
