#pragma once

#include "PyObject.hpp"

namespace py {

class PyBuiltInMethod : public PyBaseObject
{
	using FunctionType = std::function<PyResult<PyObject *>(PyTuple *, PyDict *)>;
	const std::string m_name;
	FunctionType m_builtin_method;
	PyObject *m_self;

	friend class ::Heap;

	PyBuiltInMethod(std::string name, FunctionType &&builtin_method, PyObject *self);

  public:
	static PyResult<PyBuiltInMethod *>
		create(std::string name, FunctionType &&builtin_method, PyObject *self);

	const std::string &name() { return m_name; }
	const FunctionType &builtin_method() { return m_builtin_method; }

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py