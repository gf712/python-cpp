#pragma once

#include "../forward.hpp"
#include "../utilities.hpp"
#include "forward.hpp"

#include <gmpxx.h>
#include <spdlog/fmt/bundled/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <variant>
#include <vector>

template<> struct fmt::formatter<mpz_class> : fmt::formatter<std::string>
{
	template<typename FormatContext> auto format(const mpz_class &number, FormatContext &ctx)
	{
		std::ostringstream os;
		os << number;
		return fmt::formatter<std::string>::format(os.str(), ctx);
	}
};

namespace py {

using BigIntType = mpz_class;

struct Number
{
	// Should we use `mpf_class` instead of `double`?
	std::variant<double, BigIntType> value;
	friend std::ostream &operator<<(std::ostream &os, const Number &number)
	{
		os << number.to_string();
		return os;
	}

	std::string to_string() const;

	Number exp(const Number &rhs) const;

	Number operator+(const Number &rhs) const;
	Number& operator+=(const Number &rhs);

	Number operator-(const Number &rhs) const;

	Number operator*(const Number &rhs) const;

	Number operator%(const Number &rhs) const;

	Number operator/(const Number &rhs) const;

	Number operator<<(const Number &rhs) const;
	Number operator>>(const Number &rhs) const;

	bool operator==(const Number &rhs) const;
	bool operator<=(const Number &rhs) const;
	bool operator<(const Number &rhs) const;
	bool operator>(const Number &rhs) const;
	bool operator>=(const Number &rhs) const;

	bool operator==(const PyObject *other) const;
	bool operator==(const Bytes &) const { return false; }
	bool operator==(const Ellipsis &) const { return false; }
	bool operator==(const NameConstant &) const;
	bool operator==(const Tuple &) const { return false; }

	Number floordiv(const Number &) const;
};

struct String
{
	std::string s;
	friend std::ostream &operator<<(std::ostream &os, const String &str)
	{
		os << str.to_string();
		return os;
	}

	static String from_unescaped_string(const std::string &str);

	bool operator==(const String &rhs) const { return s == rhs.s; }
	bool operator==(const PyObject *other) const;
	bool operator==(const Bytes &) const { return false; }
	bool operator==(const Ellipsis &) const { return false; }
	bool operator==(const NameConstant &) const { return false; }
	bool operator==(const Number &) const { return false; }
	bool operator==(const Tuple &) const { return false; }

	std::string to_string() const { return s; }
};

struct Bytes
{
	std::vector<std::byte> b;

	static Bytes from_unescaped_string(const std::string &);

	friend std::ostream &operator<<(std::ostream &os, const Bytes &bytes)
	{
		return os << bytes.to_string();
	}

	std::string to_string() const;

	bool operator==(const Bytes &rhs) const
	{
		return std::equal(b.begin(), b.end(), rhs.b.begin(), rhs.b.end());
	}
	bool operator==(const PyObject *other) const;
	bool operator==(const String &) const { return false; }
	bool operator==(const Ellipsis &) const { return false; }
	bool operator==(const NameConstant &) const { return false; }
	bool operator==(const Number &) const { return false; }
	bool operator==(const Tuple &) const { return false; }
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
	bool operator==(const Tuple &) const { return false; }
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
	bool operator==(const Tuple &) const { return false; }

	std::string to_string() const
	{
		return std::visit(overloaded{ [](const bool v) { return (v ? "True" : "False"); },
							  [](const NoneType) { return "None"; } },
			value);
	}
};

struct Tuple
{
	std::vector<Value> elements;

	friend std::ostream &operator<<(std::ostream &os, const Tuple &tuple);

	std::string to_string() const;

	bool operator==(const NoneType &) const { return false; }
	bool operator==(const NameConstant &) const { return false; }
	bool operator==(const PyObject *) const { return false; }
	bool operator==(const Ellipsis &) const { return false; }
	bool operator==(const Bytes &) const { return false; }
	bool operator==(const String &) const { return false; }
	bool operator==(const Number &) const { return false; }
	bool operator==(const Tuple &other) const { return elements == other.elements; }
};

class BaseException;

template<typename T> struct Ok
{
	T value;
	constexpr Ok(T value_) : value(value_) {}
};

template<typename T> Ok(T) -> Ok<T>;

struct Err
{
	BaseException *exc;
	constexpr Err(BaseException *value_) : exc(value_) {}
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
	template<typename U> constexpr PyResult(Ok<U> result_) : result(Ok<T>(result_.value))
	{
		static_assert(std::is_convertible_v<U, T>);
	}
	constexpr PyResult(Err result_) : result(result_) {}

	template<typename U> constexpr PyResult(const PyResult<U> &other) : result(Err(nullptr))
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
			std::conditional_t<detail::is_ok<typename std::invoke_result_t<FunctorType, T>>{},
				typename detail::is_ok<typename std::invoke_result_t<FunctorType, T>>::type,
				typename detail::is_pyresult<typename std::invoke_result_t<FunctorType, T>>::type>>
	PyResult<PyResultType> and_then(FunctorType &&op) const;

	template<typename FunctorType> PyResult<T> or_else(FunctorType &&op) const;
};

template<typename T>
template<typename FunctorType, typename PyResultType>
PyResult<PyResultType> PyResult<T>::and_then(FunctorType &&op) const
{
	using ResultType = typename std::invoke_result_t<FunctorType, T>;
	static_assert(detail::is_ok<ResultType>{} || detail::is_pyresult<ResultType>{},
		"Return type of function must be of type Ok<U> or PyResult<U>");
	if (is_ok()) {
		return op(unwrap());
	} else {
		return PyResult<PyResultType>(Err(unwrap_err()));
	}
}

template<typename T>
template<typename FunctorType>
PyResult<T> PyResult<T>::or_else(FunctorType &&op) const
{
	using ResultType = typename std::invoke_result_t<FunctorType, ErrType>;
	static_assert(detail::is_ok<ResultType>{} || detail::is_pyresult<ResultType>{},
		"Return type of function must be of type Ok<U> or PyResult<U>");
	if (is_err()) {
		return op(unwrap_err());
	} else {
		return PyResult<T>(Ok(unwrap()));
	}
}

PyResult<Value> add(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> subtract(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> multiply(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> exp(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> modulo(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> true_divide(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> floordiv(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> lshift(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> rshift(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> equals(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> not_equals(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> less_than_equals(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> less_than(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> greater_than(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> greater_than_equals(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> and_(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<Value> or_(const Value &lhs, const Value &rhs, Interpreter &interpreter);

PyResult<bool> is(const Value &lhs, const Value &rhs, Interpreter &interpreter);
PyResult<bool> in(const Value &lhs, const Value &rhs, Interpreter &interpreter);

PyResult<bool> truthy(const Value &lhs, Interpreter &interpreter);

bool operator==(const Value &lhs, const Value &rhs);

}// namespace py
