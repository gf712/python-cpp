#pragma once

#include "runtime/PyObject.hpp"

namespace py {

class PyWeakProxy : public PyBaseObject
{
	friend class ::Heap;
	mutable PyObject *m_object{ nullptr };
	PyObject *m_callback{ nullptr };

  protected:
	PyWeakProxy(PyType *);

	PyWeakProxy(PyObject *object, PyObject *callback);

	void visit_graph(Visitor &) override;

  public:
	static PyResult<PyWeakProxy *> create(PyObject *object, PyObject *callback);

	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __str__() const;

	static PyType *register_type(PyModule *module, std::string_view name);

  private:
	bool is_alive() const;
};

template<> PyWeakProxy *as(PyObject *obj);
template<> const PyWeakProxy *as(const PyObject *obj);

}// namespace py
