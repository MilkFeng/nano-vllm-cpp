#pragma once

#define VLLM_DEFINE_NLOHMANN_JSON_SERIALIZATION(Type) \
  void to_json(nlohmann::json& j, const Type& obj);

#define VLLM_DEFINE_NLOHMANN_JSON_DESERIALIZATION(Type) \
  void from_json(const nlohmann::json& j, Type& obj);

#define VLLM_DEFINE_NLOHMANN_JSON(Type)         \
  VLLM_DEFINE_NLOHMANN_JSON_SERIALIZATION(Type) \
  VLLM_DEFINE_NLOHMANN_JSON_DESERIALIZATION(Type)

#define VLLM_DEFINE_NLOHMANN_JSON_SERIALIZATION_FOR_ENUM(ENUM_TYPE) \
  VLLM_DEFINE_NLOHMANN_JSON_SERIALIZATION(ENUM_TYPE)

#define VLLM_DEFINE_NLOHMANN_JSON_DESERIALIZATION_FOR_ENUM(ENUM_TYPE) \
  VLLM_DEFINE_NLOHMANN_JSON_DESERIALIZATION(ENUM_TYPE)

#define VLLM_DEFINE_NLOHMANN_JSON_FOR_ENUM(ENUM_TYPE)         \
  VLLM_DEFINE_NLOHMANN_JSON_SERIALIZATION_FOR_ENUM(ENUM_TYPE) \
  VLLM_DEFINE_NLOHMANN_JSON_DESERIALIZATION_FOR_ENUM(ENUM_TYPE)

#define VLLM_IMPL_NLOHMANN_JSON_SERIALIZATION_FOR_ENUM(ENUM_TYPE, ...)         \
  void to_json(nlohmann::json& j, const ENUM_TYPE& e) {                        \
    static_assert(std::is_enum<ENUM_TYPE>::value,                              \
                  #ENUM_TYPE " must be an enum!");                             \
    static const std::pair<ENUM_TYPE, const nlohmann::json> m[] = __VA_ARGS__; \
    auto it = std::find_if(                                                    \
        std::begin(m), std::end(m),                                            \
        [e](const std::pair<ENUM_TYPE, const nlohmann::json>& ej_pair)         \
            -> bool { return ej_pair.first == e; });                           \
    if (it == std::end(m)) {                                                   \
      throw std::invalid_argument("invalid value for enum " #ENUM_TYPE);       \
    }                                                                          \
    j = it->second;                                                            \
  }

#define VLLM_IMPL_NLOHMANN_JSON_DESERIALIZATION_FOR_ENUM(ENUM_TYPE, ...)       \
  void from_json(const nlohmann::json& j, ENUM_TYPE& e) {                      \
    static_assert(std::is_enum<ENUM_TYPE>::value,                              \
                  #ENUM_TYPE " must be an enum!");                             \
    static const std::pair<ENUM_TYPE, const nlohmann::json> m[] = __VA_ARGS__; \
    auto it = std::find_if(                                                    \
        std::begin(m), std::end(m),                                            \
        [&j](const std::pair<ENUM_TYPE, const nlohmann::json>& ej_pair)        \
            -> bool { return ej_pair.second == j; });                          \
    if (it == std::end(m)) {                                                   \
      throw std::invalid_argument("unknown JSON value for enum " #ENUM_TYPE);  \
    }                                                                          \
    e = it->first;                                                             \
  }

#define VLLM_IMPL_NLOHMANN_JSON_FOR_ENUM(ENUM_TYPE, ...)                 \
  VLLM_IMPL_NLOHMANN_JSON_SERIALIZATION_FOR_ENUM(ENUM_TYPE, __VA_ARGS__) \
  VLLM_IMPL_NLOHMANN_JSON_DESERIALIZATION_FOR_ENUM(ENUM_TYPE, __VA_ARGS__)

#define VLLM_IMPL_NLOHMANN_JSON_SERIALIZATION(Type)  \
  void to_json(nlohmann::json& j, const Type& obj) { \
    j = obj.to_json();                               \
  }

#define VLLM_IMPL_NLOHMANN_JSON_DESERIALIZATION(Type)  \
  void from_json(const nlohmann::json& j, Type& obj) { \
    obj = Type::from_json(j);                          \
  }

#define VLLM_IMPL_NLOHMANN_JSON(Type)         \
  VLLM_IMPL_NLOHMANN_JSON_SERIALIZATION(Type) \
  VLLM_IMPL_NLOHMANN_JSON_DESERIALIZATION(Type)
