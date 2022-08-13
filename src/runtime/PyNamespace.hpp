#pragma once

#include "PyObject.hpp"

namespace py {

class PyNamespace : public PyBaseObject
{
	friend class ::Heap;

  public:
	PyDict *m_dict{ nullptr };

  private:
	PyNamespace(PyDict *);

  protected:
	void visit_graph(Visitor &) override;

  public:
	static PyResult<PyNamespace *> create();
	static PyResult<PyNamespace *> create(PyDict *dict);

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __eq__(const PyObject *obj) const;
	PyResult<PyObject *> __lt__(const PyObject *obj) const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
	std::string to_string() const override;
};

}// namespace py