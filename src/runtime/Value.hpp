#pragma once

#include "../forward.hpp"
#include "forward.hpp"
#include "utilities.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <sstream>
#include <variant>
#include <vector>

namespace py {

struct Number
{
	std::variant<int64_t, double> value;
	friend std::ostream &operator<<(std::ostream &os, const Number &number)
	{
		os << number.to_string();
		return os;
	}

	std::string to_string() const
	{
		return std::visit([](const auto &value) { return std::to_string(value); }, value);
	}

	Number exp(const Number &rhs) const
	{
		return std::visit(
			[](const auto &lhs_value, const auto &rhs_value) {
				return Number{ static_cast<decltype(lhs_value)>(
					std::pow(static_cast<double>(lhs_value), static_cast<double>(rhs_value))) };
			},
			value,
			rhs.value);
	}

	Number operator+(const Number &rhs) const
	{
		return std::visit(
			[](const auto &lhs_value, const auto &rhs_value) {
				return Number{ static_cast<decltype(lhs_value)>(
					static_cast<double>(lhs_value) + static_cast<double>(rhs_value)) };
			},
			value,
			rhs.value);
	}
	Number operator-(const Number &rhs) const
	{
		return std::visit(
			[](const auto &lhs_value, const auto &rhs_value) {
				return Number{ static_cast<decltype(lhs_value)>(
					static_cast<double>(lhs_value) - static_cast<double>(rhs_value)) };
			},
			value,
			rhs.value);
	}

	Number operator*(const Number &rhs) const;

	Number operator%(const Number &rhs) const
	{
		return std::visit(
			[](const auto &lhs_value, const auto &rhs_value) {
				return Number{ static_cast<decltype(lhs_value)>(
					std::fmod(static_cast<double>(lhs_value), static_cast<double>(rhs_value))) };
			},
			value,
			rhs.value);
	}
	Number operator/(const Number &rhs) const;
	Number operator<<(const Number &rhs) const
	{
		return std::visit(
			[](const auto &lhs_value, const auto &rhs_value) {
				// FIXME
				return Number{ static_cast<int64_t>(lhs_value) << static_cast<int64_t>(rhs_value) };
			},
			value,
			rhs.value);
	}

	bool operator==(const Number &rhs) const;
	bool operator<=(const Number &rhs) const;
	bool operator<(const Number &rhs) const;
	bool operator>(const Number &rhs) const;
	bool operator>=(const Number &rhs) const;

	bool operator==(const PyObject *other) const;
	bool operator==(const Bytes &) const { return false; }
	bool operator==(const Ellipsis &) const { return false; }
	bool operator==(const NameConstant &) const;
};

struct String
{
	std::string s;
	friend std::ostream &operator<<(std::ostream &os, const String &s)
	{
		os << s.to_string();
		return os;
	}

	bool operator==(const String &rhs) const { return s == rhs.s; }
	bool operator==(const PyObject *other) const;
	bool operator==(const Bytes &) const { return false; }
	bool operator==(const Ellipsis &) const { return false; }
	bool operator==(const NameConstant &) const { return false; }
	bool operator==(const Number &) const { return false; }

	std::string to_string() const { return s; }
};

struct Bytes
{
	std::vector<std::byte> b;
	friend std::ostream &operator<<(std::ostream &os, const Bytes &bytes)
	{
		return os << bytes.to_string();
	}

	std::string to_string() const
	{
		std::ostringstream os;
		for (const auto &byte_ : b) { os << std::to_integer<uint8_t>(byte_); }
		return os.str();
	}

	bool operator==(const Bytes &rhs) const
	{
		return std::equal(b.begin(), b.end(), rhs.b.begin(), rhs.b.end());
	}
	bool operator==(const PyObject *other) const;
	bool operator==(const String &) const { return false; }
	bool operator==(const Ellipsis &) const { return false; }
	bool operator==(const NameConstant &) const { return false; }
	bool operator==(const Number &) const { return false; }
};

struct Ellipsis
{
	static constexpr std::string_view ellipsis_repr = "...";
	friend std::ostream &operator<<(std::ostream &os, const Ellipsis &)
	{
		return os << Ellipsis::ellipsis_repr;
	}

	std::string to_string() const { return std::string(ellipsis_repr); }

	bool operator==(const Ellipsis &) const { return true; }
	bool operator==(const PyObject *other) const;
	bool operator==(const Bytes &) const { return false; }
	bool operator==(const String &) const { return false; }
	bool operator==(const Number &) const { return false; }
};

struct NoneType
{
	friend std::ostream &operator<<(std::ostream &os, const NoneType &) { return os << "None"; }

	bool operator==(const NoneType &) const { return true; }

	template<typename T> bool operator==(const T &) const { return false; }

	std::string to_string() const { return "None"; }
	bool operator==(const PyObject *other) const;
};

struct NameConstant
{
	std::variant<bool, NoneType> value;
	friend std::ostream &operator<<(std::ostream &os, const NameConstant &val)
	{
		return os << val.to_string();
	}

	bool operator==(const NoneType &) const { return false; }
	bool operator==(const NameConstant &other) const;
	bool operator==(const PyObject *other) const;
	bool operator==(const Ellipsis &) const { return false; }
	bool operator==(const Bytes &) const { return false; }
	bool operator==(const String &) const { return false; }
	bool operator==(const Number &) const;

	std::string to_string() const
	{
		return std::visit(overloaded{ [](const bool value) { return (value ? "True" : "False"); },
							  [](const NoneType) { return "None"; } },
			value);
	}
};

class BaseException;

template<typename T> struct Ok
{
	T value;
	Ok(T value_) : value(value_) {}
};

template<typename T> Ok(T) -> Ok<T>;

struct Err
{
	BaseException *exc;
	Err(BaseException *value_) : exc(value_) {}
};

template<typename T> class PyResult;

namespace detail {
	template<typename> struct is_ok : std::false_type
	{
	};

	template<typename T> struct is_ok<Ok<T>> : std::true_type
	{
		using type = T;
	};

	template<typename> struct is_pyresult : std::false_type
	{
	};

	template<typename T> struct is_pyresult<PyResult<T>> : std::true_type
	{
		using type = T;
	};
}// namespace detail


template<typename T> class PyResult
{
  public:
	using OkType = T;
	using ErrType = BaseException *;
	using StorageType = std::variant<Ok<T>, Err>;

  private:
	StorageType result;

  public:
	PyResult(Ok<T> result_) : result(result_) {}
	template<typename U> PyResult(Ok<U> result_) : result(Ok<T>(result_.value))
	{
		static_assert(std::is_convertible_v<U, T>);
	}
	PyResult(Err result_) : result(result_) {}

	template<typename U> PyResult(const PyResult<U> &other) : result(Err(nullptr))
	{
		static_assert(std::is_convertible_v<U, T>);
		if (other.is_ok()) {
			result = Ok<T>(other.unwrap());
		} else {
			result = Err(other.unwrap_err());
		}
	}

	bool is_ok() const { return std::holds_alternative<Ok<T>>(result); }
	bool is_err() const { return !is_ok(); }

	T unwrap() const
	{
		ASSERT(is_ok());
		return std::get<Ok<T>>(result).value;
	}

	BaseException *unwrap_err() const
	{
		ASSERT(is_err());
		return std::get<Err>(result).exc;
	}

	template<typename FunctorType,
		typename PyResultType =
			std::conditional_t<detail::is_ok<typename std::result_of_t<FunctorType(T)>>{},
				typename detail::is_ok<typename std::result_of_t<FunctorType(T)>>::type,
				typename detail::is_pyresult<typename std::result_of_t<FunctorType(T)>>::type>>
	PyResult<PyResultType> and_then(FunctorType &&op) const;
};

template<typename T>
template<typename FunctorType, typename PyResultType>
PyResult<PyResultType> PyResult<T>::and_then(FunctorType &&op) const
{
	using ResultType = typename std::result_of_t<FunctorType(T)>;
	static_assert(detail::is_ok<ResultType>{} || detail::is_pyresult<ResultType>{},
		"Return type of function must be of type Ok<U> or PyResult<U>");
	if (is_ok()) {
		return op(unwrap());
	} else {
		return PyResult<PyResultType>(Err(unwrap_err()));
	}
}


PyResult<Value> add(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> subtract(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> multiply(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> exp(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> modulo(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> true_divide(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> lshift(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> equals(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> not_equals(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> less_than_equals(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> less_than(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> greater_than(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> greater_than_equals(const Value &lhs, const Value &rhs, Interpreter &interpreter);

PyResult<bool> is(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<bool> in(const Value &lhs, const Value &rhs, Interpreter &interpreter);

PyResult<bool> truthy(const Value &lhs, Interpreter &interpreter);

}// namespace py