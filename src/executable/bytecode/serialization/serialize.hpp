#pragma once

#include "runtime/PyTuple.hpp"

#include <string>
#include <vector>

namespace py {

template<typename T> struct is_vector : std::false_type
{
	static constexpr bool value = false;
};

template<typename T> struct is_vector<std::vector<T>> : std::true_type
{
	using ElementType = T;
	static constexpr bool value = true;
};

template<typename T> inline constexpr bool is_vector_v = is_vector<T>::value;

template<typename T>
concept Vector = is_vector_v<T>;

enum class ValueType {
	INT64 = 0,
	F64 = 1,
	STRING = 2,
	BYTES = 3,
	ELLIPSIS = 4,
	NONE = 5,
	BOOL = 6,
	OBJECT = 7,
};

template<typename T> inline void serialize(const T &value, std::vector<uint8_t> &result)
{
	result.reserve(result.size() + sizeof(T));
	for (size_t i = 0; i < sizeof(T); ++i) {
		result.push_back(reinterpret_cast<const uint8_t *>(&value)[i]);
	}
}

template<> inline void serialize<bool>(const bool &value, std::vector<uint8_t> &result)
{
	result.push_back(value ? 1 : 0);
}

template<> inline void serialize<std::byte>(const std::byte &value, std::vector<uint8_t> &result)
{
	result.push_back(::bit_cast<uint8_t>(value));
}

template<>
inline void serialize<std::string>(const std::string &value, std::vector<uint8_t> &result)
{
	serialize(value.size(), result);
	for (const auto &el : value) { serialize(el, result); }
}

template<Vector VectorType>
inline void serialize(const VectorType &value, std::vector<uint8_t> &result)
{
	serialize(value.size(), result);
	for (const auto &el : value) { serialize(el, result); }
}

template<> inline void serialize<PyTuple *>(PyTuple *const &value, std::vector<uint8_t> &result)
{
	serialize(value->size(), result);
	for (const auto &el : value->elements()) {
		std::visit(
			overloaded{
				[&](const Number &val) {
					std::visit(overloaded{ [&](const int64_t &v) {
											  serialize(
												  static_cast<uint8_t>(ValueType::INT64), result);
											  serialize(v, result);
										  },
								   [&](const double &v) {
									   serialize(static_cast<uint8_t>(ValueType::F64), result);
									   serialize(v, result);
								   } },
						val.value);
				},
				[&](const String &val) {
					serialize(static_cast<uint8_t>(ValueType::STRING), result);
					serialize(val.s, result);
				},
				[&](const Bytes &bytes) {
					serialize(static_cast<uint8_t>(ValueType::BYTES), result);
					serialize(bytes.b, result);
				},
				[&](const Ellipsis &) {
					serialize(static_cast<uint8_t>(ValueType::ELLIPSIS), result);
				},
				[&](const NameConstant &val) {
					std::visit(overloaded{ [&](const bool &v) {
											  serialize(
												  static_cast<uint8_t>(ValueType::BOOL), result);
											  serialize(v, result);
										  },
								   [&](const NoneType &) {
									   serialize(static_cast<uint8_t>(ValueType::NONE), result);
								   } },
						val.value);
				},
				[&](PyObject *const &) { TODO(); },
			},
			el);
	}
}
}// namespace py