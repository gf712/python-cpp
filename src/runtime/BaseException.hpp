#pragma once

#include "PyObject.hpp"


class BaseException : public PyBaseObject
{
	PyTuple *m_args;

  protected:
	BaseException(const TypePrototype &type, PyTuple *args);

	static PyType *s_base_exception_type;

  public:
	BaseException(PyTuple *args);

	std::string what() const;

	std::string to_string() const override;

	static PyType *register_type(PyModule *);

	PyType *type() const override;

	void visit_graph(Visitor &) override;
};