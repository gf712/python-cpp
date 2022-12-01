#pragma once

#include "PyObject.hpp"

namespace py {
class PyZip : public PyBaseObject
{
	friend class ::Heap;

	std::vector<PyObject *> m_iterators;

  private:
	PyZip(std::vector<PyObject *> &&iterators);

  public:
	static PyResult<PyObject *> create(PyTuple *iterables);
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __next__();

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};
}// namespace py
