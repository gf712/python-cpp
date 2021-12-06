#pragma once

#include "PyObject.hpp"

class PySlotWrapper : public PyBaseObject
{
	PyString *m_name;
	PyType *m_underlying_type;
	std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> m_slot;

	friend class Heap;

	PySlotWrapper(PyString *name,
		PyType *underlying_type,
		std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> function);

  public:
	static PySlotWrapper *create(PyString *name,
		PyType *underlying_type,
		std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> function);

	PyString *slot_name() { return m_name; }
	const std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> &slot() { return m_slot; }

	std::string to_string() const override;

	PyObject *__repr__() const;
	PyObject *__call__(PyTuple *args, PyDict *kwargs);

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type_() const override;
};

template<> inline PySlotWrapper *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_SLOT_WRAPPER) {
		return static_cast<PySlotWrapper *>(node);
	}
	return nullptr;
}

template<> inline const PySlotWrapper *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_SLOT_WRAPPER) {
		return static_cast<const PySlotWrapper *>(node);
	}
	return nullptr;
}