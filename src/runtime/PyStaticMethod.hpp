#pragma once

#include "PyObject.hpp"

namespace py {

class PyStaticMethod : public PyBaseObject
{
	PyString *m_name;
	PyType *m_underlying_type{ nullptr };
	PyObject *m_static_method;

	friend class ::Heap;

	PyStaticMethod(PyString *name, PyType *underlying_type, PyObject *function);

  public:
	static PyStaticMethod *create(PyString *name, PyObject *function);

	PyString *static_method_name() { return m_name; }
	PyObject *static_method() { return m_static_method; }

	std::string to_string() const override;

	PyObject *__repr__() const;
	PyObject *__get__(PyObject *instance, PyObject *owner) const;
	PyObject *call_static_method(PyTuple *args, PyDict *kwargs);

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py