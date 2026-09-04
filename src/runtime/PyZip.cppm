module;
#include "memory/allocate.hpp"

export module py.runtime:zip;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyTuple;
class PyDict;
class PyType;
class PyZip : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;

	std::vector<PyObject *> m_iterators;

  private:
	PyZip(PyType *type);

	PyZip(std::vector<PyObject *> &&iterators);

  public:
	static PyResult<PyObject *> create(PyTuple *iterables);
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __next__();

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};
}// namespace py
