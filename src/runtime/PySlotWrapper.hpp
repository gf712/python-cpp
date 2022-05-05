#pragma once

#include "PyObject.hpp"

namespace py {

class PySlotWrapper : public PyBaseObject
{
	using FunctionType = std::function<PyResult<PyObject *>(PyObject *, PyTuple *, PyDict *)>;
	PyString *m_name;
	PyType *m_slot_type;
	FunctionType m_slot;

	friend class ::Heap;

	PySlotWrapper(PyString *name, PyType *underlying_type, FunctionType &&function);

  public:
	static PyResult<PySlotWrapper *>
		create(PyString *name, PyType *underlying_type, FunctionType &&function);

	PyString *slot_name() { return m_name; }
	const FunctionType &slot() { return m_slot; }

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __get__(PyObject *, PyObject *) const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py