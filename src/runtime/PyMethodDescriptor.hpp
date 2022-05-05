#pragma once

#include "MemoryError.hpp"
#include "PyObject.hpp"
#include "vm/VM.hpp"

namespace py {

class PyMethodDescriptor : public PyBaseObject
{
	using FunctionType = std::function<PyResult<PyObject *>(PyObject *, PyTuple *, PyDict *)>;
	PyString *m_name;
	PyType *m_underlying_type;
	FunctionType m_method_descriptor;
	std::vector<PyObject *> m_captures;

	friend class ::Heap;

	PyMethodDescriptor(PyString *name,
		PyType *underlying_type,
		FunctionType &&function,
		std::vector<PyObject *> &&captures);

  public:
	template<typename... Args>
	static PyResult<PyMethodDescriptor *>
		create(PyString *name, PyType *underlying_type, FunctionType &&function, Args &&... args)
	{
		auto *obj = VirtualMachine::the().heap().allocate<PyMethodDescriptor>(
			name, underlying_type, std::move(function), std::vector<PyObject *>{ args... });
		if (!obj) { return Err(memory_error(sizeof(PyMethodDescriptor))); }
		return Ok(obj);
	}

	PyString *name() { return m_name; }

	const FunctionType &method_descriptor() { return m_method_descriptor; }

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __get__(PyObject *, PyObject *) const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py