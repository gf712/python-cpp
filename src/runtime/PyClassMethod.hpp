#pragma once

#include "PyObject.hpp"

namespace py {

class PyClassMethod : public PyBaseObject
{
	friend class ::Heap;

  public:
	PyObject *m_callable;

  private:
	PyClassMethod(PyType *);
	PyClassMethod();

  public:
	static PyResult<PyClassMethod *> create();

	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __get__(PyObject *instance, PyObject *owner) const;

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
