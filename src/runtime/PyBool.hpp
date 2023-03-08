#pragma once

#include "PyInteger.hpp"

namespace py {

class PyBool : public PyInteger
{
	friend class ::Heap;
	friend PyObject *py_true();
	friend PyObject *py_false();

	PyBool(PyType *);

  public:
	std::string to_string() const override;

	bool value() const;

	void visit_graph(Visitor &) override {}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<bool> __bool__() const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

  private:
	static PyResult<PyBool *> create(bool);

	PyBool(bool name);
};

PyObject *py_true();
PyObject *py_false();

}// namespace py
