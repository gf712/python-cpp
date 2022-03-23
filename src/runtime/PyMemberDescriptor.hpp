#pragma once

#include "PyObject.hpp"
#include "vm/VM.hpp"

namespace py {

class PyMemberDescriptor : public PyBaseObject
{
	PyString *m_name;
	PyType *m_underlying_type;
	std::function<PyObject *(PyObject *)> m_member_accessor;

	friend class ::Heap;

	PyMemberDescriptor(PyString *name,
		PyType *underlying_type,
		std::function<PyObject *(PyObject *)> member);

  public:
	static PyMemberDescriptor *create(PyString *name,
		PyType *underlying_type,
		std::function<PyObject *(PyObject *)> member)
	{
		return VirtualMachine::the().heap().allocate<PyMemberDescriptor>(
			name, underlying_type, member);
	}

	PyString *name() { return m_name; }

	std::string to_string() const override;

	PyObject *__repr__() const;
	PyObject *__get__(PyObject *, PyObject *) const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py