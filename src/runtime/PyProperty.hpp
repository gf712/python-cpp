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
	static PyProperty *create(PyObject *fget, PyObject *fset, PyObject *fdel, PyString *);

	std::string to_string() const override;

	static PyObject *__new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyObject *__repr__() const;
	PyObject *__get__(PyObject *instance, PyObject *owner) const;

	PyObject *getter(PyTuple *args, PyDict *kwargs) const;
	PyObject *setter(PyTuple *args, PyDict *kwargs) const;
	PyObject *deleter(PyTuple *args, PyDict *kwargs) const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py