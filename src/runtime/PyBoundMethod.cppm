module;
#include "memory/allocate.hpp"

export module py.runtime:bound_method;
import :dict;
import :function;
import :object;
import :tuple;
import :value;
import py.memory;
import std;

export namespace py {
class PyType;

class PyBoundMethod : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	PyObject *m_self;
	PyFunction *m_method;

	PyBoundMethod(PyObject *self, PyFunction *method);

	PyBoundMethod(PyType *);

  public:
	static PyResult<PyBoundMethod *> create(PyObject *self, PyFunction *method);

	PyObject *self() { return m_self; }
	PyFunction *method() { return m_method; }

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
