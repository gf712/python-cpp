#pragma once

#include "PyObject.hpp"

class PyNumber : public PyBaseObject
{
	friend class Heap;
	friend class PyInteger;
	friend class PyFloat;

	Number m_value;

  public:
	static PyNumber *create(const Number &number);
	std::string to_string() const override
	{
		return std::visit(
			[](const auto &value) { return fmt::format("{}", value); }, m_value.value);
	}

	PyObject *__add__(const PyObject *obj) const;
	PyObject *__sub__(const PyObject *obj) const;
	PyObject *__mod__(const PyObject *obj) const;
	PyObject *__mul__(const PyObject *obj) const;

	PyObject *__abs__() const;
	PyObject *__neg__() const;
	PyObject *__pos__() const;
	PyObject *__invert__() const;

	PyObject *__repr__() const;
	PyObject *__eq__(const PyObject *obj) const;
	PyObject *__ne__(const PyObject *obj) const;
	PyObject *__lt__(const PyObject *obj) const;
	PyObject *__le__(const PyObject *obj) const;
	PyObject *__gt__(const PyObject *obj) const;
	PyObject *__ge__(const PyObject *obj) const;

	const Number &value() const { return m_value; }

	static const PyNumber *as_number(const PyObject *obj);

  private:
	PyNumber(Number number, const TypePrototype &type) : PyBaseObject(type), m_value(number) {}
};
