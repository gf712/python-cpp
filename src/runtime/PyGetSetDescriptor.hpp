#pragma once

#include "PyObject.hpp"

namespace py {

class PyGetSetDescriptor : public PyBaseObject
{
	PyString *m_name;
	PyType *m_underlying_type;
	PropertyDefinition &m_getset;

	friend class ::Heap;

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
	PyType *type() const override;
};

}// namespace py