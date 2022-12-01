#pragma once

#include "PyObject.hpp"

namespace py {
class PyReversed : public PyBaseObject
{
	friend class ::Heap;

	PyObject *m_sequence{ nullptr };

  private:
	PyReversed(PyObject *sequence);

  public:
    // can return an object that is not PyReversed, if the sequence implements __reversed__
	static PyResult<PyObject *> create(PyObject *sequence);
    static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __next__();

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};
}// namespace py
