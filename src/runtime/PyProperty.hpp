#pragma once

#include "PyObject.hpp"

namespace py {

class PyProperty : public PyBaseObject
{
	PyObject *m_getter;
	PyObject *m_setter;
	PyObject *m_deleter;
	PyString *m_property_name;

	friend class ::Heap;

	PyProperty(PyObject *fget, PyObject *fset, PyObject *fdel, PyString *);

  public:
	static PyResult create(PyObject *fget, PyObject *fset, PyObject *fdel, PyString *);

	std::string to_string() const override;

	static PyResult __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult __repr__() const;
	PyResult __get__(PyObject *instance, PyObject *owner) const;

	PyResult getter(PyTuple *args, PyDict *kwargs) const;
	PyResult setter(PyTuple *args, PyDict *kwargs) const;
	PyResult deleter(PyTuple *args, PyDict *kwargs) const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py