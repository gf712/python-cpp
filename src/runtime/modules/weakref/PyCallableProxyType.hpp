#pragma once

#include "runtime/PyObject.hpp"

namespace py {

class PyCallableProxyType : public PyBaseObject
{
	friend class ::Heap;
	mutable PyObject *m_object{ nullptr };
	PyObject *m_callback{ nullptr };

  protected:
	PyCallableProxyType(PyType *);

	PyCallableProxyType(PyObject *object, PyObject *callback);

	void visit_graph(Visitor &) override;

  public:
	static PyResult<PyCallableProxyType *> create(PyObject *object, PyObject *callback);

	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __str__() const;
	PyResult<PyObject *> __getattribute__(PyObject *attribute) const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);

	static PyType *register_type(PyModule *module, std::string_view name);

  private:
	bool is_alive() const;
};

template<> PyCallableProxyType *as(PyObject *obj);
template<> const PyCallableProxyType *as(const PyObject *obj);

}// namespace py
