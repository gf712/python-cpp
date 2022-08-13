#pragma once

#include "PyObject.hpp"

namespace py {

class PyBuiltInMethod : public PyBaseObject
{
	MethodDefinition &m_ml;
	PyObject *m_self;

	friend class ::Heap;

	PyBuiltInMethod(MethodDefinition &method_definition, PyObject *self);

  public:
	static PyResult<PyBuiltInMethod *> create(MethodDefinition &method_definition, PyObject *self);

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};

}// namespace py