#pragma once

#include "PyObject.hpp"

namespace py {

class PySlotWrapper : public PyBaseObject
{
	PyString *m_name;
	PyType *m_slot_type;
	std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> m_slot;

	friend class ::Heap;

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
	PyObject *__get__(PyObject *, PyObject *) const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py