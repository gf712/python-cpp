#pragma once

#include "PyObject.hpp"
#include "vm/VM.hpp"

class PyMethodWrapper : public PyBaseObject
{
	PyString *m_name;
	PyType *m_underlying_type;
	std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> m_method_descriptor;
	std::vector<PyObject *> m_captures;

	friend class Heap;

	PyMethodWrapper(PyString *name,
		PyType *underlying_type,
		std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> function,
		std::vector<PyObject *> &&captures);

  public:
	template<typename... Args>
	static PyMethodWrapper *create(PyString *name,
		PyType *underlying_type,
		std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> function,
		Args &&... args)
	{
		return VirtualMachine::the().heap().allocate<PyMethodWrapper>(
			name, underlying_type, function, std::vector<PyObject *>{ args... });
	}

	PyString *name() { return m_name; }
	const std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> &method_descriptor()
	{
		return m_method_descriptor;
	}

	std::string to_string() const override;

	PyObject *__repr__() const;
	PyObject *__call__(PyTuple *args, PyDict *kwargs);

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};
