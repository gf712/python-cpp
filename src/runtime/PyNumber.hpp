#pragma once

#include "PyObject.hpp"

namespace py {

class PyNumber : public PyBaseObject
{
	friend class ::Heap;
	friend Interface<PyNumber, PyInteger>;
	friend class PyFloat;

  protected:
	Number m_value;

  public:
	static PyResult<PyNumber *> create(const Number &number);
	std::string to_string() const override;

	PyResult<PyObject *> __add__(const PyObject *obj) const;
	PyResult<PyObject *> __sub__(const PyObject *obj) const;
	PyResult<PyObject *> __mod__(const PyObject *obj) const;
	PyResult<PyObject *> __mul__(const PyObject *obj) const;

	PyResult<PyObject *> __abs__() const;
	PyResult<PyObject *> __neg__() const;
	PyResult<PyObject *> __pos__() const;
	PyResult<PyObject *> __invert__() const;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __eq__(const PyObject *obj) const;
	PyResult<PyObject *> __ne__(const PyObject *obj) const;
	PyResult<PyObject *> __lt__(const PyObject *obj) const;
	PyResult<PyObject *> __le__(const PyObject *obj) const;
	PyResult<PyObject *> __gt__(const PyObject *obj) const;
	PyResult<PyObject *> __ge__(const PyObject *obj) const;

	PyResult<bool> __bool__() const;

	const Number &value() const { return m_value; }

	static const PyNumber *as_number(const PyObject *obj);

  private:
	PyNumber(Number number, const TypePrototype &type) : PyBaseObject(type), m_value(number) {}
};

}// namespace py