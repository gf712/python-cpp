#pragma once

#include "PyObject.hpp"

namespace py {

class PySuper : public PyBaseObject
{
	friend ::Heap;

	PyType *m_type{ nullptr };
	PyObject *m_object{ nullptr };
	PyType *m_object_type{ nullptr };

	PySuper(PyType *);

	PySuper();
	PySuper(PyType *type, PyObject *object, PyType *object_type);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __getattribute__(PyObject *name) const;
	PyResult<PyObject *> __get__(PyObject *object, PyObject *type) const;

	std::string to_string() const override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;

  private:
	static PyResult<PyType *> check(PyType *type, PyObject *object);

	static PyResult<PyObject *> infer_object(PyFrame *, PyCode *);

	static PyResult<PyType *> infer_type(PyFrame *, PyCode *);

	void visit_graph(Visitor &) override;
};

}// namespace py
