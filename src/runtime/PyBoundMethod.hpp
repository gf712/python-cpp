#pragma once

#include "PyObject.hpp"

class PyBoundMethod : public PyBaseObject
{
	PyObject *m_self;
	PyFunction *m_method;

  public:
	PyBoundMethod(PyObject *self, PyFunction *method);

	PyObject *self() { return m_self; }
	PyFunction *method() { return m_method; }

	std::string to_string() const override;

	PyObject *__repr__() const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type_() const override;
};

template<> inline PyBoundMethod *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_BOUND_METHOD) {
		return static_cast<PyBoundMethod *>(node);
	}
	return nullptr;
}

template<> inline const PyBoundMethod *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_BOUND_METHOD) {
		return static_cast<const PyBoundMethod *>(node);
	}
	return nullptr;
}