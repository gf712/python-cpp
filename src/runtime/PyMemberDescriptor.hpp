#pragma once

#include "PyObject.hpp"
#include "vm/VM.hpp"

namespace py {

class PyMemberDescriptor : public PyBaseObject
{
	PyString *m_name;
	PyType *m_underlying_type;
	std::function<PyResult<PyObject *>(PyObject *)> m_member_accessor;
	std::function<PyResult<std::monostate>(PyObject *, PyObject *)> m_member_setter;

	friend class ::Heap;

	PyMemberDescriptor(PyType *);

	PyMemberDescriptor(PyString *name,
		PyType *underlying_type,
		std::function<PyResult<PyObject *>(PyObject *)> member,
		std::function<PyResult<std::monostate>(PyObject *, PyObject *)> member_setter);

  public:
	static PyResult<PyMemberDescriptor *> create(PyString *name,
		PyType *underlying_type,
		std::function<PyResult<PyObject *>(PyObject *)> member,
		std::function<PyResult<std::monostate>(PyObject *, PyObject *)> member_setter);

	PyString *name() { return m_name; }

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __get__(PyObject *, PyObject *) const;
	PyResult<std::monostate> __set__(PyObject *obj, PyObject *value);

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
