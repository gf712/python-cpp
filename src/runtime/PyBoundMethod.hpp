#pragma once

#include "PyObject.hpp"

namespace py {

class PyBoundMethod : public PyBaseObject
{
	friend class ::Heap;
	PyObject *m_self;
	PyFunction *m_method;

	PyBoundMethod(PyObject *self, PyFunction *method);

  public:
	static PyResult create(PyObject *self, PyFunction *method);

	PyObject *self() { return m_self; }
	PyFunction *method() { return m_method; }

	std::string to_string() const override;

	PyResult __repr__() const;
	PyResult __call__(PyTuple *args, PyDict *kwargs);

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py