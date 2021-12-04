#pragma once

#include "PyObject.hpp"

class CustomPyObject : public PyBaseObject
{
	const PyType *m_type_obj;

  public:
	CustomPyObject(const PyType *type);

	std::string to_string() const override { return fmt::format("object"); }

	static PyObject *__new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	std::optional<int32_t> __init__(PyTuple *args, PyDict *kwargs);

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type_() const override;
};