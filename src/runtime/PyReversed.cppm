module;
#include "memory/allocate.hpp"

export module py.runtime:reversed;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyTuple;
class PyDict;
class PyType;
class PyReversed : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;

	PyObject *m_sequence{ nullptr };

  private:
	PyReversed(PyType *type);

	PyReversed(PyObject *sequence);

  public:
	// can return an object that is not PyReversed, if the sequence implements __reversed__
	static PyResult<PyObject *> create(PyObject *sequence);
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __next__();

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};
}// namespace py
