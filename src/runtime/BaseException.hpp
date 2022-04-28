#pragma once

#include "PyObject.hpp"

namespace py {

class BaseException : public PyBaseObject
{
	friend BaseException *as<>(PyObject *obj);
	friend const BaseException *as<>(const PyObject *obj);

  protected:
	PyTuple *m_args;

	BaseException(const TypePrototype &type, PyTuple *args);

	static PyType *s_base_exception_type;

  public:
	BaseException(PyTuple *args);

	std::string what() const;
	PyResult __repr__() const;

	std::string to_string() const override;

	static PyType *register_type(PyModule *);

	PyType *type() const override;
	static PyType *static_type();

	void visit_graph(Visitor &) override;
};

}// namespace py