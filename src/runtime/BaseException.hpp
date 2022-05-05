#pragma once

#include "PyObject.hpp"

namespace py {

class BaseException : public PyBaseObject
{
	friend class ::Heap;
	template<typename T> friend struct klass;

	friend BaseException *as<>(PyObject *obj);
	friend const BaseException *as<>(const PyObject *obj);

  protected:
	PyTuple *m_args{ nullptr };
	PyTraceback *m_traceback{ nullptr };

	BaseException(const TypePrototype &type, PyTuple *args);

	static PyType *s_base_exception_type;

	BaseException(PyTuple *args);

  public:
	static PyResult<BaseException *> create(PyTuple *args);

	std::string what() const;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;

	std::string to_string() const override;

	PyTuple *args() const { return m_args; }

	PyTraceback *traceback() const { return m_traceback; }
	void set_traceback(PyTraceback *tb) { m_traceback = tb; }

	std::string format_traceback() const;

	static PyType *register_type(PyModule *);

	PyType *type() const override;
	static PyType *static_type();

	void visit_graph(Visitor &) override;
};

}// namespace py