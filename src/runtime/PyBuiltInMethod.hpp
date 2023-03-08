#pragma once

#include "PyObject.hpp"

namespace py {

class PyBuiltInMethod : public PyBaseObject
{
	friend class ::Heap;

	std::optional<std::reference_wrapper<MethodDefinition>> m_ml;
	PyObject *m_self;

	PyBuiltInMethod(PyType *);

	PyBuiltInMethod(MethodDefinition &method_definition, PyObject *self);

  public:
	static PyResult<PyBuiltInMethod *> create(MethodDefinition &method_definition, PyObject *self);

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
