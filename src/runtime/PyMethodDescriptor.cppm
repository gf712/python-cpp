module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:method_descriptor;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyTuple;
class PyString;
class PyDict;
class PyType;

class PyMethodDescriptor : public PyBaseObject
{
	PyString *m_name{ nullptr };
	PyType *m_underlying_type{ nullptr };
	std::optional<std::reference_wrapper<MethodDefinition>> m_method;
	std::vector<PyObject *> m_captures;

	friend class ::Heap;
	friend class py::detail::Allocator;

	PyMethodDescriptor(PyType *);

	PyMethodDescriptor(PyString *name,
		PyType *underlying_type,
		MethodDefinition &method,
		std::vector<PyObject *> &&captures);

  public:
	static PyResult<PyMethodDescriptor *> create(PyString *name,
		PyType *underlying_type,
		MethodDefinition &method,
		std::vector<PyObject *> &&captures);

	template<typename... Args>
	static PyResult<PyMethodDescriptor *>
		create(PyString *name, PyType *underlying_type, MethodDefinition &method, Args &&...args)
	{
		return PyMethodDescriptor::create(
			name, underlying_type, method, std::vector<PyObject *>{ args... });
	}

	PyString *name() { return m_name; }

	const MethodDefinition &method_descriptor() const
	{
		ASSERT(m_method);
		return m_method->get();
	}

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __get__(PyObject *, PyObject *) const;

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
