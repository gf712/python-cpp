#pragma once

#include "MemoryError.hpp"
#include "PyObject.hpp"
#include "vm/VM.hpp"

namespace py {

class PyClassMethodDescriptor : public PyBaseObject
{
	friend class ::Heap;

	PyString *m_name;
	PyType *m_underlying_type;
	std::optional<std::reference_wrapper<MethodDefinition>> m_method;
	std::vector<PyObject *> m_captures;

	PyClassMethodDescriptor(PyType *);

	PyClassMethodDescriptor(PyString *name,
		PyType *underlying_type,
		MethodDefinition &method,
		std::vector<PyObject *> &&captures);

  public:
	template<typename... Args>
	static PyResult<PyClassMethodDescriptor *>
		create(PyString *name, PyType *underlying_type, MethodDefinition &method, Args &&...args)
	{
		auto *obj = VirtualMachine::the().heap().allocate<PyClassMethodDescriptor>(
			name, underlying_type, method, std::vector<PyObject *>{ args... });
		if (!obj) { return Err(memory_error(sizeof(PyClassMethodDescriptor))); }
		return Ok(obj);
	}

	PyString *name() { return m_name; }

	const MethodDefinition &method_descriptor() const
	{
		ASSERT(m_method);
		return m_method->get();
	}

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __get__(PyObject *, PyObject *) const;

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
