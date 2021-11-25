#include "Value.hpp"
#include "PyBool.hpp"
#include "PyBytes.hpp"
#include "PyEllipsis.hpp"
#include "PyFloat.hpp"
#include "PyInteger.hpp"
#include "PyNone.hpp"
#include "PyNumber.hpp"
#include "PyString.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"


bool Number::operator==(const PyObject *other) const
{
	if (auto other_pynumber = as<PyInteger>(other)) {
		return *this == other_pynumber->value();
	} else if (auto other_pynumber = as<PyFloat>(other)) {
		return *this == other_pynumber->value();
	} else {
		return false;
	}
}


bool Number::operator==(const NameConstant &other) const
{
	if (std::holds_alternative<NoneType>(other.value)) { return false; }
	if (*this == Number{ int64_t{ 0 } }) { return std::get<bool>(other.value) == false; }
	if (*this == Number{ int64_t{ 1 } }) { return std::get<bool>(other.value) == true; }
	return false;
}

bool Number::operator==(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) { return lhs_value == rhs_value; },
		value,
		rhs.value);
}
bool Number::operator<=(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) { return lhs_value <= rhs_value; },
		value,
		rhs.value);
}
bool Number::operator<(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) { return lhs_value < rhs_value; },
		value,
		rhs.value);
}

bool Number::operator>(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) { return lhs_value > rhs_value; },
		value,
		rhs.value);
}

bool Number::operator>=(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) { return lhs_value >= rhs_value; },
		value,
		rhs.value);
}


bool String::operator==(const PyObject *other) const
{
	if (auto other_pystr = as<PyString>(other)) {
		return s == other_pystr->value();
	} else {
		return false;
	}
}


bool Bytes::operator==(const PyObject *other) const
{
	if (auto other_pybytes = as<PyBytes>(other)) {
		return *this == other_pybytes->value();
	} else {
		return false;
	}
}

bool Ellipsis::operator==(const PyObject *other) const { return other == py_ellipsis(); }


bool NoneType::operator==(const PyObject *other) const { return other == py_none(); }


bool NameConstant::operator==(const PyObject *other) const
{
	if (std::holds_alternative<NoneType>(value)) { return other == py_none(); }
	const auto bool_value = std::get<bool>(value);
	if (bool_value) {
		return other == py_true();
	} else {
		return other == py_false();
	}
}


bool NameConstant::operator==(const Number &other) const
{
	if (std::holds_alternative<NoneType>(value)) { return false; }
	const short bool_value = std::get<bool>(value);
	if (bool_value) {
		return other == Number{ int64_t{ 1 } };
	} else {
		return other == Number{ int64_t{ 0 } };
	}
}

std::optional<Value> add(const Value &lhs, const Value &rhs, Interpreter &)
{
	auto result = std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value + rhs_value;
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->add(py_rhs)) { return result; }
				return {};
			} },
		lhs,
		rhs);

	return result;
}

std::optional<Value> subtract(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value - rhs_value;
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->subtract(py_rhs)) { return result; }
				return {};
			} },
		lhs,
		rhs);
}

std::optional<Value> multiply(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value * rhs_value;
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->multiply(py_rhs)) { return result; }
				return {};
			} },
		lhs,
		rhs);
}

std::optional<Value> exp(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value.exp(rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->exp(py_rhs)) { return result; }
				return {};
			} },
		lhs,
		rhs);
}

std::optional<Value> lshift(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value << rhs_value;
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->lshift(py_rhs)) { return result; }
				return {};
			} },
		lhs,
		rhs);
}

std::optional<Value> modulo(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value % rhs_value;
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->modulo(py_rhs)) { return result; }
				return {};
			} },
		lhs,
		rhs);
}

std::optional<Value> equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> std::optional<Value> {
					   return NameConstant{ lhs_value == rhs_value };
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value == rhs_value };
			},
			[](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value == rhs_value };
			},
			[](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->richcompare(py_rhs, RichCompare::Py_EQ)) {
					return result;
				} else {
					return {};
				}
			} },
		lhs,
		rhs);
}

std::optional<Value> less_than_equals(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> std::optional<Value> {
					   return NameConstant{ lhs_value <= rhs_value };
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value <= rhs_value };
			},
			[](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value <= rhs_value };
			},
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->richcompare(py_rhs, RichCompare::Py_LE)) {
					return result;
				} else {
					interpreter.raise_exception(
						"TypeError: unsupported operand type(s) for <=: \'{}\' and \'{}\'",
						object_name(py_lhs->type()),
						object_name(py_rhs->type()));
					return {};
				}
			} },
		lhs,
		rhs);
}


std::optional<Value> less_than(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> std::optional<Value> {
					   return NameConstant{ lhs_value < rhs_value };
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value < rhs_value };
			},
			[](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value < rhs_value };
			},
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->richcompare(py_rhs, RichCompare::Py_LT)) {
					return result;
				} else {
					interpreter.raise_exception(
						"TypeError: unsupported operand type(s) for <: \'{}\' and \'{}\'",
						object_name(py_lhs->type()),
						object_name(py_rhs->type()));
					return {};
				}
			} },
		lhs,
		rhs);
}

bool truthy(const Value &value, Interpreter &)
{
	// Number, String, Bytes, Ellipsis, NameConstant, PyObject *
	return std::visit(overloaded{ [](const NameConstant &c) {
									 if (std::holds_alternative<NoneType>(c.value)) {
										 return false;
									 } else {
										 return std::get<bool>(c.value);
									 }
								 },
						  [](const Number &number) { return number == Number{ int64_t{ 0 } }; },
						  [](const String &str) { return !str.s.empty(); },
						  [](const Bytes &bytes) { return !bytes.b.empty(); },
						  [](const Ellipsis &) { return true; },
						  [](PyObject *obj) { return obj->bool_() == py_true(); } },
		value);
}
