#pragma once

#include "PyObject.hpp"

namespace py {

class PyNumber : public PyBaseObject
{
	friend class ::Heap;
	friend class PyInteger;
	friend class PyFloat;

	Number m_value;

  public:
	static PyResult create(const Number &number);
	std::string to_string() const override;

	PyResult __add__(const PyObject *obj) const;
	PyResult __sub__(const PyObject *obj) const;
	PyResult __mod__(const PyObject *obj) const;
	PyResult __mul__(const PyObject *obj) const;

	PyResult __abs__() const;
	PyResult __neg__() const;
	PyResult __pos__() const;
	PyResult __invert__() const;

	PyResult __repr__() const;
	PyResult __eq__(const PyObject *obj) const;
	PyResult __ne__(const PyObject *obj) const;
	PyResult __lt__(const PyObject *obj) const;
	PyResult __le__(const PyObject *obj) const;
	PyResult __gt__(const PyObject *obj) const;
	PyResult __ge__(const PyObject *obj) const;

	const Number &value() const { return m_value; }

	static const PyNumber *as_number(const PyObject *obj);

  private:
	PyNumber(Number number, const TypePrototype &type) : PyBaseObject(type), m_value(number) {}
};

}// namespace py