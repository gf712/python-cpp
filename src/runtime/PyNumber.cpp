#include "PyNumber.hpp"

#include "interpreter/Interpreter.hpp"

std::shared_ptr<PyObject> PyNumber::equal_impl(const std::shared_ptr<PyObject> &obj,
	Interpreter &) const
{
	if (auto pynum = as<PyNumber>(obj)) {
		const bool comparisson = m_value == pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	return nullptr;
}


std::shared_ptr<PyObject> PyNumber::add_impl(const std::shared_ptr<PyObject> &obj,
	Interpreter &interpreter) const
{
	if (auto rhs = as<PyNumber>(obj)) {
		return PyNumber::create(m_value + rhs->value());
	} else {
		interpreter.raise_exception(
			"TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
			object_name(type()),
			object_name(obj->type()));
		return nullptr;
	}
}


std::shared_ptr<PyObject> PyNumber::subtract_impl(const std::shared_ptr<PyObject> &obj,
	Interpreter &interpreter) const
{
	if (auto rhs = as<PyNumber>(obj)) {
		return PyNumber::create(m_value - rhs->value());
	} else {
		interpreter.raise_exception(
			"TypeError: unsupported operand type(s) for -: \'{}\' and \'{}\'",
			object_name(type()),
			object_name(obj->type()));
		return nullptr;
	}
}


std::shared_ptr<PyObject> PyNumber::modulo_impl(const std::shared_ptr<PyObject> &obj,
	Interpreter &interpreter) const
{
	if (auto rhs = as<PyNumber>(obj)) {
		return PyNumber::create(m_value % rhs->value());
	} else {
		interpreter.raise_exception(
			"TypeError: unsupported operand type(s) for %: \'{}\' and \'{}\'",
			object_name(type()),
			object_name(obj->type()));
		return nullptr;
	}
}

std::shared_ptr<PyObject> PyNumber::multiply_impl(const std::shared_ptr<PyObject> &obj,
	Interpreter &interpreter) const
{
	if (auto rhs = as<PyNumber>(obj)) {
		return PyNumber::create(m_value * rhs->value());
	} else {
		interpreter.raise_exception(
			"TypeError: unsupported operand type(s) for *: \'{}\' and \'{}\'",
			object_name(type()),
			object_name(obj->type()));
		return nullptr;
	}
}
