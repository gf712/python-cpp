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
	static PyResult<PyProperty *>
		create(PyObject *fget, PyObject *fset, PyObject *fdel, PyString *);

	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __get__(PyObject *instance, PyObject *owner) const;

	PyResult<PyObject *> getter(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> setter(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> deleter(PyTuple *args, PyDict *kwargs) const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py