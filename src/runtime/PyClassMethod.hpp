#pragma once

#include "PyObject.hpp"

namespace py {

class PyClassMethod : public PyBaseObject
{
	friend class ::Heap;

	PyObject *m_callable;

	PyClassMethod();

  public:
	static PyResult create();

	std::string to_string() const override;

	static PyResult __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	std::optional<int32_t> __init__(PyTuple *args, PyDict *kwargs);
	PyResult __repr__() const;
	PyResult __get__(PyObject *instance, PyObject *owner) const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py