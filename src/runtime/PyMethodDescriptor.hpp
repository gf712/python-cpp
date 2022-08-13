#pragma once

#include "MemoryError.hpp"
#include "PyObject.hpp"
#include "vm/VM.hpp"

namespace py {

class PyMethodDescriptor : public PyBaseObject
{
	PyString *m_name;
	PyType *m_underlying_type;
	MethodDefinition &m_method;
	std::vector<PyObject *> m_captures;

	friend class ::Heap;

	PyMethodDescriptor(PyString *name,
		PyType *underlying_type,
		MethodDefinition &method,
		std::vector<PyObject *> &&captures);

  public:
	template<typename... Args>
	static PyResult<PyMethodDescriptor *>
		create(PyString *name, PyType *underlying_type, MethodDefinition &method, Args &&...args)
	{
		auto *obj = VirtualMachine::the().heap().allocate<PyMethodDescriptor>(
			name, underlying_type, method, std::vector<PyObject *>{ args... });
		if (!obj) { return Err(memory_error(sizeof(PyMethodDescriptor))); }
		return Ok(obj);
	}

	PyString *name() { return m_name; }

	const MethodDefinition &method_descriptor() const { return m_method; }

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __get__(PyObject *, PyObject *) const;

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};

}// namespace py