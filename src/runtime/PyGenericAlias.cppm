module;
#include "memory/allocate.hpp"

export module py.runtime:generic_alias;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyTuple;
class PyType;
class PyGenericAlias : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;

	PyObject *m_origin{ nullptr };
	PyTuple *m_args{ nullptr };
	PyObject *m_parameters{ nullptr };

	PyGenericAlias(PyType *type);

	PyGenericAlias(PyObject *origin, PyTuple *args, PyObject *parameters);

  public:
	static PyResult<PyGenericAlias *>
		create(PyObject *origin, PyObject *args, PyObject *parameters);

	static PyResult<PyGenericAlias *> create(PyObject *origin, PyObject *args);

	PyResult<PyObject *> __repr__() const;
	std::string to_string() const override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
	void visit_graph(Visitor &visitor) override;
};
}// namespace py
