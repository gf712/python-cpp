#pragma once

#include "PyObject.hpp"

namespace py {

class PyClassMethod : public PyBaseObject
{
	friend class ::Heap;

	PyObject *m_callable;

	PyClassMethod();

  public:
	static PyClassMethod *create();

	std::string to_string() const override;

	static PyObject *__new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	std::optional<int32_t> __init__(PyTuple *args, PyDict *kwargs);
	PyObject *__repr__() const;
	PyObject *__get__(PyObject *instance, PyObject *owner) const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py