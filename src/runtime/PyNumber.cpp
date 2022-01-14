#include "PyNumber.hpp"
#include "PyFloat.hpp"
#include "PyInteger.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"

#include "interpreter/Interpreter.hpp"

using namespace py;

std::string PyNumber::to_string() const
{
	return std::visit(overloaded{
						  [](const double &value) { return fmt::format("{:f}", value); },
						  [](const int64_t &value) { return fmt::format("{}", value); },
					  },
		m_value.value);
}

PyObject *PyNumber::__repr__() const { return PyString::create(to_string()); }

const PyNumber *PyNumber::as_number(const PyObject *obj)
{
	if (auto *num = as<PyFloat>(obj)) {
		return num;
	} else if (auto *num = as<PyInteger>(obj)) {
		return num;
	}
	return nullptr;
}

PyObject *PyNumber::__abs__() const
{
	return PyNumber::create(
		std::visit([](const auto &val) { return Number{ std::abs(val) }; }, m_value.value));
}

PyObject *PyNumber::__neg__() const
{
	return PyNumber::create(
		std::visit([](const auto &val) { return Number{ -val }; }, m_value.value));
}

PyObject *PyNumber::__pos__() const
{
	return PyNumber::create(
		std::visit([](const auto &val) { return Number{ val }; }, m_value.value));
}

PyObject *PyNumber::__invert__() const
{
	if (std::get<double>(m_value.value)) {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("bad operand type for unary ~: 'float'"));
		return nullptr;
	}
	return PyNumber::create(Number{ ~std::get<int64_t>(m_value.value) });
}

PyObject *PyNumber::__add__(const PyObject *obj) const
{
	if (auto *rhs = as_number(obj)) {
		return PyNumber::create(m_value + rhs->value());
	} else {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("unsupported operand type(s) for +: \'{}\' and \'{}\'",
				type()->name(),
				obj->type()->name()));
		return nullptr;
	}
}

PyObject *PyNumber::__sub__(const PyObject *obj) const
{
	if (auto rhs = as_number(obj)) {
		return PyNumber::create(m_value - rhs->value());
	} else {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("unsupported operand type(s) for -: \'{}\' and \'{}\'",
				type()->name(),
				obj->type()->name()));
		return nullptr;
	}
}

PyObject *PyNumber::__mod__(const PyObject *obj) const
{
	if (auto rhs = as_number(obj)) {
		return PyNumber::create(m_value % rhs->value());
	} else {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("unsupported operand type(s) for %: \'{}\' and \'{}\'",
				type()->name(),
				obj->type()->name()));
		return nullptr;
	}
}

PyObject *PyNumber::__mul__(const PyObject *obj) const
{
	if (auto rhs = as_number(obj)) {
		return PyNumber::create(m_value * rhs->value());
	} else {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("unsupported operand type(s) for *: \'{}\' and \'{}\'",
				type()->name(),
				obj->type()->name()));
		return nullptr;
	}
}

PyObject *PyNumber::__eq__(const PyObject *obj) const
{
	if (auto *pynum = as_number(obj)) {
		const bool comparisson = m_value == pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("'==' not supported between instances of '{}' and '{}'",
			type()->name(),
			obj->type()->name()));
	return nullptr;
}


PyObject *PyNumber::__ne__(const PyObject *obj) const
{
	if (auto *pynum = as_number(obj)) {
		const bool comparisson = m_value != pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("'!=' not supported between instances of '{}' and '{}'",
			type()->name(),
			obj->type()->name()));
	return nullptr;
}


PyObject *PyNumber::__lt__(const PyObject *other) const
{
	if (auto *pynum = as_number(other)) {
		const bool comparisson = m_value < pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("'<' not supported between instances of '{}' and '{}'",
			type()->name(),
			other->type()->name()));
	return nullptr;
}

PyObject *PyNumber::__le__(const PyObject *other) const
{
	if (auto *pynum = as_number(other)) {
		const bool comparisson = m_value <= pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("'<=' not supported between instances of '{}' and '{}'",
			type()->name(),
			other->type()->name()));
	return nullptr;
}

PyObject *PyNumber::__gt__(const PyObject *other) const
{
	if (auto *pynum = as_number(other)) {
		const bool comparisson = m_value > pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("'>' not supported between instances of '{}' and '{}'",
			type()->name(),
			other->type()->name()));
	return nullptr;
}

PyObject *PyNumber::__ge__(const PyObject *other) const
{
	if (auto *pynum = as_number(other)) {
		const bool comparisson = m_value >= pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("'>=' not supported between instances of '{}' and '{}'",
			type()->name(),
			other->type()->name()));
	return nullptr;
}

PyNumber *PyNumber::create(const Number &number)
{
	if (std::holds_alternative<double>(number.value)) {
		return PyFloat::create(std::get<double>(number.value));
	} else {
		return PyInteger::create(std::get<int64_t>(number.value));
	}
}
