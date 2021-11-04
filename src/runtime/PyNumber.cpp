#include "PyNumber.hpp"
#include "TypeError.hpp"

#include "interpreter/Interpreter.hpp"

PyObject *PyNumber::add_impl(const PyObject *obj, Interpreter &interpreter) const
{
	if (auto *rhs = as<PyNumber>(obj)) {
		return PyNumber::create(m_value + rhs->value());
	} else {
		interpreter.raise_exception(
			"TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
			object_name(type()),
			object_name(obj->type()));
		return nullptr;
	}
}


PyObject *PyNumber::subtract_impl(const PyObject *obj, Interpreter &interpreter) const
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


PyObject *PyNumber::modulo_impl(const PyObject *obj, Interpreter &interpreter) const
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

PyObject *PyNumber::multiply_impl(const PyObject *obj, Interpreter &interpreter) const
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


PyObject *PyNumber::equal_impl(const PyObject *obj, Interpreter &) const
{
	if (auto *pynum = as<PyNumber>(obj)) {
		const bool comparisson = m_value == pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	type_error("'==' not supported between instances of '{}' and '{}'",
		object_name(this->type()),
		object_name(obj->type()));
	return nullptr;
}


PyObject *PyNumber::less_than_impl(const PyObject *other, Interpreter &) const
{
	if (auto *pynum = as<PyNumber>(other)) {
		const bool comparisson = m_value < pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	type_error("'<' not supported between instances of '{}' and '{}'",
		object_name(this->type()),
		object_name(other->type()));
	return nullptr;
}

PyObject *PyNumber::less_than_equal_impl(const PyObject *other, Interpreter &) const
{
	if (auto *pynum = as<PyNumber>(other)) {
		const bool comparisson = m_value <= pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	type_error("'<=' not supported between instances of '{}' and '{}'",
		object_name(this->type()),
		object_name(other->type()));
	return nullptr;
}

PyObject *PyNumber::greater_than_impl(const PyObject *other, Interpreter &) const
{
	if (auto *pynum = as<PyNumber>(other)) {
		const bool comparisson = m_value > pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	type_error("'>' not supported between instances of '{}' and '{}'",
		object_name(this->type()),
		object_name(other->type()));
	return nullptr;
}

PyObject *PyNumber::greater_than_equal_impl(const PyObject *other, Interpreter &) const
{
	if (auto *pynum = as<PyNumber>(other)) {
		const bool comparisson = m_value >= pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	type_error("'>=' not supported between instances of '{}' and '{}'",
		object_name(this->type()),
		object_name(other->type()));
	return nullptr;
}