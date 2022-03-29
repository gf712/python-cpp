#pragma once

#include "runtime/Value.hpp"
#include "serialize.hpp"

#include <span>
#include <string>

namespace py {

template<typename T> inline T deserialize(std::span<const uint8_t> &buffer)
{
	if constexpr (std::is_same_v<T, std::string>) {
		const size_t string_size = deserialize<size_t>(buffer);
		std::string result;
		result.resize(string_size);
		for (size_t i = 0; i < string_size; ++i) {
			result[i] = *reinterpret_cast<const char *>(&buffer[i]);
		}
		buffer = buffer.subspan(string_size);
		return result;
	} else if constexpr (std::is_same_v<T, size_t>) {
		size_t result;
		for (size_t i = 0; i < sizeof(size_t); ++i) {
			reinterpret_cast<uint8_t *>(&result)[i] = buffer[i];
		}
		buffer = buffer.subspan(sizeof(size_t));
		return result;
	} else if constexpr (std::is_same_v<T, double>) {
		double result;
		for (size_t i = 0; i < sizeof(double); ++i) {
			reinterpret_cast<uint8_t *>(&result)[i] = buffer[i];
		}
		buffer = buffer.subspan(sizeof(double));
		return result;
	} else if constexpr (std::is_same_v<T, int64_t>) {
		int64_t result;
		for (size_t i = 0; i < sizeof(int64_t); ++i) {
			reinterpret_cast<uint8_t *>(&result)[i] = buffer[i];
		}
		buffer = buffer.subspan(sizeof(int64_t));
		return result;
	} else if constexpr (std::is_same_v<T, uint8_t>) {
		uint8_t result = buffer.front();
		buffer = buffer.subspan(sizeof(uint8_t));
		return result;
	} else if constexpr (std::is_same_v<T, bool>) {
		bool result = buffer.front() == uint8_t{ 1 };
		buffer = buffer.subspan(sizeof(uint8_t));
		return result;
	} else if constexpr (is_vector_v<T>) {
		const size_t vector_size = deserialize<size_t>(buffer);
		T result;
		result.resize(vector_size);
		for (size_t i = 0; i < vector_size; ++i) {
			result[i] = deserialize<typename is_vector<T>::ElementType>(buffer);
		}
		return result;
	} else if constexpr (std::is_same_v<T, PyTuple *>) {
		const size_t vector_size = deserialize<size_t>(buffer);
		std::vector<Value> result;
		result.resize(vector_size);
		for (size_t i = 0; i < vector_size; ++i) { result[i] = deserialize<Value>(buffer); }
		return PyTuple::create(result);
	} else if constexpr (std::is_same_v<T, Value>) {
		const uint8_t type = deserialize<uint8_t>(buffer);
		switch (static_cast<ValueType>(type)) {
		case ValueType::INT64: {
			return Number{ deserialize<int64_t>(buffer) };
		} break;
		case ValueType::F64: {
			return Number{ deserialize<double>(buffer) };
		} break;
		case ValueType::STRING: {
			return String{ deserialize<std::string>(buffer) };
		} break;
		case ValueType::BYTES: {
			TODO();
		} break;
		case ValueType::ELLIPSIS: {
			return Ellipsis{};
		} break;
		case ValueType::NONE: {
			return NameConstant{ NoneType{} };
		} break;
		case ValueType::BOOL: {
			return NameConstant{ deserialize<bool>(buffer) };
		} break;
		case ValueType::OBJECT: {
			TODO();
		} break;
		}
	} else {
		[]<bool flag = false>() { static_assert(flag, "unsupported deserialization type"); }
		();
	}
}
}// namespace py