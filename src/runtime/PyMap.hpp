#pragma once

#include "PyObject.hpp"

namespace py {

class PyMap : public PyBaseObject
{
	friend class ::Heap;

	PyObject *m_func{ nullptr };
	PyTuple *m_iters{ nullptr };

	PyMap(PyType *);
	PyMap(PyType *, PyObject *func, PyTuple *iters);

  public:
	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __next__();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
	void visit_graph(Visitor &) override;
};

}// namespace py
