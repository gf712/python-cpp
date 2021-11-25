#pragma once

#include "PyObject.hpp"

class PyBuiltInMethod : public PyBaseObject
{
	const std::string m_name;
	std::function<PyObject *(PyTuple *, PyDict *)> m_builtin_method;
	PyObject *m_self;

	friend class Heap;

	PyBuiltInMethod(std::string name,
		std::function<PyObject *(PyTuple *, PyDict *)> builtin_method,
		PyObject *self);

  public:
	static PyBuiltInMethod *create(std::string name,
		std::function<PyObject *(PyTuple *, PyDict *)> builtin_method,
		PyObject *self);

	const std::string &name() { return m_name; }
	const std::function<PyObject *(PyTuple *, PyDict *)> &builtin_method()
	{
		return m_builtin_method;
	}

	std::string to_string() const override;

	PyObject *__repr__() const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type_() const override;
};

template<> inline PyBuiltInMethod *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_BUILTIN_METHOD) {
		return static_cast<PyBuiltInMethod *>(node);
	}
	return nullptr;
}

template<> inline const PyBuiltInMethod *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_BUILTIN_METHOD) {
		return static_cast<const PyBuiltInMethod *>(node);
	}
	return nullptr;
}