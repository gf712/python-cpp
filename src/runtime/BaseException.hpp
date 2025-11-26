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
	PyObject *m_dict{ nullptr };
	PyTraceback *m_traceback{ nullptr };
	PyObject *m_context{ nullptr };
	PyObject *m_cause{ nullptr };
	bool m_suppress_context{ false };

	BaseException(PyType *type);

	BaseException(PyType *type, PyTuple *args);

	BaseException(const TypePrototype &type, PyTuple *args);

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

	PyObject *cause() const { return m_cause; }
	void set_cause(PyObject *cause) { m_cause = cause; }

	std::string format_traceback() const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
	static PyType *class_type();

	void visit_graph(Visitor &) override;
};

}// namespace py
