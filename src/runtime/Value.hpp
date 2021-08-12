#pragma once

#include "forward.hpp"
#include "utilities.hpp"

#include <variant>
#include <sstream>
#include <vector>
#include <memory>
#include <cstddef>
#include <cmath>

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

	Number operator*(const Number &rhs) const
	{
		return std::visit(
			[](const auto &lhs_value, const auto &rhs_value) {
				return Number{ static_cast<decltype(lhs_value)>(
					static_cast<double>(lhs_value) * static_cast<double>(rhs_value)) };
			},
			value,
			rhs.value);
	}

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
	bool operator==(const Number &rhs) const
	{
		return std::visit(
			[](const auto &lhs_value, const auto &rhs_value) { return lhs_value == rhs_value; },
			value,
			rhs.value);
	}
	bool operator==(const std::shared_ptr<PyObject> &other) const;
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
	bool operator==(const std::shared_ptr<PyObject> &other) const;
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
	bool operator==(const std::shared_ptr<PyObject> &other) const;
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
	bool operator==(const std::shared_ptr<PyObject> &other) const;
	bool operator==(const Bytes &) const { return false; }
	bool operator==(const String &) const { return false; }
	bool operator==(const Number &) const { return false; }
};

struct NoneType
{
	friend std::ostream &operator<<(std::ostream &os, const NoneType &) { return os << "None"; }

	bool operator==(const NoneType &) const { return true; }

	template<typename T> bool operator==(T &&) const { return false; }

	std::string to_string() const { return "None"; }
	bool operator==(const std::shared_ptr<PyObject> &other) const;
};

struct NameConstant
{
	std::variant<bool, NoneType> value;
	friend std::ostream &operator<<(std::ostream &os, const NameConstant &val)
	{
		return os << val.to_string();
	}

	bool operator==(const NoneType &) const { return false; }

	bool operator==(const NameConstant &other) const
	{
		return std::visit(
			[](const auto &rhs, const auto &lhs) { return rhs == lhs; }, value, other.value);
	}
	bool operator==(const std::shared_ptr<PyObject> &other) const;
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

using Value =
	std::variant<Number, String, Bytes, Ellipsis, NameConstant, std::shared_ptr<PyObject>>;
