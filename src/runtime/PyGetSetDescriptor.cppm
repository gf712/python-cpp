module;
#include "memory/allocate.hpp"

export module py.runtime:get_set_descriptor;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyString;
class PyType;

class PyGetSetDescriptor : public PyBaseObject
{
	PyString *m_name;
	PyType *m_underlying_type;
	std::optional<std::reference_wrapper<PropertyDefinition>> m_getset;

	friend class ::Heap;
	friend class py::detail::Allocator;

	PyGetSetDescriptor(PyType *);

	PyGetSetDescriptor(PyString *name, PyType *underlying_type, PropertyDefinition &getset);

  public:
	static PyResult<PyGetSetDescriptor *>
		create(PyString *name, PyType *underlying_type, PropertyDefinition &getset);

	PyString *name() { return m_name; }

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __get__(PyObject *, PyObject *) const;
	PyResult<std::monostate> __set__(PyObject *obj, PyObject *value);

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
