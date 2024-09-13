#pragma once

#include "runtime/PyObject.hpp"

namespace py {

class PyWeakRef : public PyBaseObject
{
	friend class ::Heap;
	mutable PyObject *m_object{ nullptr };
	PyObject *m_callback{ nullptr };

  protected:
	PyWeakRef(PyType *);

	PyWeakRef(PyObject *object, PyObject *callback);

	void visit_graph(Visitor &) override;

  public:
	static PyResult<PyWeakRef *> create(PyObject *object, PyObject *callback);

	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs) const;

	static PyType *register_type(PyModule *module, std::string_view name);

  public:
	PyObject *get_object() const { return m_object; }

  private:
	bool is_alive() const;
};

template<> PyWeakRef *as(PyObject *obj);
template<> const PyWeakRef *as(const PyObject *obj);

}// namespace py
