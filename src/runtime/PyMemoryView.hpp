#pragma once

#include "PyObject.hpp"

namespace py {

class PyMemoryView : public PyBaseObject
{
	friend class ::Heap;

	PyObject *m_object{ nullptr };

	PyMemoryView(PyType *);
	PyMemoryView(PyType *type, PyObject *object);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

	void visit_graph(Visitor &) override;
	std::string to_string() const override;
};

}// namespace py
