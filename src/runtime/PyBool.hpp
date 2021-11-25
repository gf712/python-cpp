#pragma once

#include "PyObject.hpp"


class PyBool : public PyBaseObject
{
	friend class Heap;
	friend PyObject *py_true();
	friend PyObject *py_false();

	bool m_value;

  public:
	std::string to_string() const override;

	PyObject *add_impl(const PyObject *obj) const;

	PyObject *repr_impl() const;

	bool value() const { return m_value; }

	void visit_graph(Visitor &) override {}

	PyObject *__bool__() const;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type_() const override;

  private:
	static PyBool *create(bool);

	PyBool(bool name);
};

PyObject *py_true();
PyObject *py_false();

template<> inline PyBool *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_BOOL) { return static_cast<PyBool *>(node); }
	return nullptr;
}

template<> inline const PyBool *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_BOOL) { return static_cast<const PyBool *>(node); }
	return nullptr;
}