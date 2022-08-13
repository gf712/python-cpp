#include "PyNumber.hpp"
#include "PyBool.hpp"
#include "PyFloat.hpp"
#include "PyInteger.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "types/builtin.hpp"

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

PyResult<PyObject *> PyNumber::__repr__() const { return PyString::create(to_string()); }

const PyNumber *PyNumber::as_number(const PyObject *obj)
{
	if (obj->type()->issubclass(py::float_())) {
		return static_cast<const PyFloat *>(obj);
	} else if (obj->type()->issubclass(py::integer())) {
		return static_cast<const PyInteger *>(obj);
	}
	return nullptr;
}

PyResult<PyObject *> PyNumber::__abs__() const
{
	return PyNumber::create(
		std::visit([](const auto &val) { return Number{ std::abs(val) }; }, m_value.value));
}

PyResult<PyObject *> PyNumber::__neg__() const
{
	return PyNumber::create(
		std::visit([](const auto &val) { return Number{ -val }; }, m_value.value));
}

PyResult<PyObject *> PyNumber::__pos__() const
{
	return PyNumber::create(
		std::visit([](const auto &val) { return Number{ val }; }, m_value.value));
}

PyResult<PyObject *> PyNumber::__invert__() const
{
	if (std::get<double>(m_value.value)) {
		return Err(type_error("bad operand type for unary ~: 'float'"));
	}
	return PyNumber::create(Number{ ~std::get<int64_t>(m_value.value) });
}

PyResult<PyObject *> PyNumber::__add__(const PyObject *obj) const
{
	if (auto *rhs = as_number(obj)) {
		return PyNumber::create(m_value + rhs->value());
	} else {
		return Err(type_error("unsupported operand type(s) for +: \'{}\' and \'{}\'",
			type()->name(),
			obj->type()->name()));
	}
}

PyResult<PyObject *> PyNumber::__sub__(const PyObject *obj) const
{
	if (auto rhs = as_number(obj)) {
		return PyNumber::create(m_value - rhs->value());
	} else {
		return Err(type_error("unsupported operand type(s) for -: \'{}\' and \'{}\'",
			type()->name(),
			obj->type()->name()));
	}
}

PyResult<PyObject *> PyNumber::__mod__(const PyObject *obj) const
{
	if (auto rhs = as_number(obj)) {
		return PyNumber::create(m_value % rhs->value());
	} else {
		return Err(type_error("unsupported operand type(s) for %: \'{}\' and \'{}\'",
			type()->name(),
			obj->type()->name()));
	}
}

PyResult<PyObject *> PyNumber::__mul__(const PyObject *obj) const
{
	if (auto rhs = as_number(obj)) {
		return PyNumber::create(m_value * rhs->value());
	} else {
		return Err(type_error("unsupported operand type(s) for *: \'{}\' and \'{}\'",
			type()->name(),
			obj->type()->name()));
	}
}

PyResult<PyObject *> PyNumber::__eq__(const PyObject *obj) const
{
	if (auto *pynum = as_number(obj)) {
		const bool comparisson = m_value == pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	return Err(type_error("'==' not supported between instances of '{}' and '{}'",
		type()->name(),
		obj->type()->name()));
}


PyResult<PyObject *> PyNumber::__ne__(const PyObject *obj) const
{
	if (auto *pynum = as_number(obj)) {
		const bool comparisson = m_value != pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	return Err(type_error("'!=' not supported between instances of '{}' and '{}'",
		type()->name(),
		obj->type()->name()));
}


PyResult<PyObject *> PyNumber::__lt__(const PyObject *other) const
{
	if (auto *pynum = as_number(other)) {
		const bool comparisson = m_value < pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	return Err(type_error("'<' not supported between instances of '{}' and '{}'",
		type()->name(),
		other->type()->name()));
}

PyResult<PyObject *> PyNumber::__le__(const PyObject *other) const
{
	if (auto *pynum = as_number(other)) {
		const bool comparisson = m_value <= pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	return Err(type_error("'<=' not supported between instances of '{}' and '{}'",
		type()->name(),
		other->type()->name()));
}

PyResult<PyObject *> PyNumber::__gt__(const PyObject *other) const
{
	if (auto *pynum = as_number(other)) {
		const bool comparisson = m_value > pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	return Err(type_error("'>' not supported between instances of '{}' and '{}'",
		type()->name(),
		other->type()->name()));
}

PyResult<PyObject *> PyNumber::__ge__(const PyObject *other) const
{
	if (auto *pynum = as_number(other)) {
		const bool comparisson = m_value >= pynum->value();
		return PyObject::from(NameConstant{ comparisson });
	}
	return Err(type_error("'>=' not supported between instances of '{}' and '{}'",
		type()->name(),
		other->type()->name()));
}

PyResult<bool> PyNumber::__bool__() const
{
	if (std::holds_alternative<double>(m_value.value)) {
		return Ok(std::fpclassify(std::get<double>(m_value.value)) != FP_ZERO);
	} else {
		return Ok(std::get<int64_t>(m_value.value) != int64_t{ 0 });
	}
}

PyResult<PyNumber *> PyNumber::create(const Number &number)
{
	if (std::holds_alternative<double>(number.value)) {
		return PyFloat::create(std::get<double>(number.value));
	} else {
		return PyInteger::create(std::get<int64_t>(number.value));
	}
}
