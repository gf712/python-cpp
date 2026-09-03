module;
#include "memory/allocate.hpp"

export module py.runtime:static_method;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyTuple;
class PyDict;
class PyType;

class PyStaticMethod : public PyBaseObject
{
	PyType *m_underlying_type{ nullptr };
	PyObject *m_static_method{ nullptr };

	friend class ::Heap;
	friend class py::detail::Allocator;

	PyStaticMethod(PyType *);

	PyStaticMethod(PyType *underlying_type, PyObject *function);

  public:
	static PyResult<PyStaticMethod *> create(PyObject *function);

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyObject *static_method() { return m_static_method; }

	PyResult<PyObject *> __get__(PyObject *instance, PyObject *owner) const;
	PyResult<PyObject *> call_static_method(PyTuple *args, PyDict *kwargs);

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
