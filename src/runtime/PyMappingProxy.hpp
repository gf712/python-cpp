#pragma once

#include "PyObject.hpp"

namespace py {
class PyMappingProxy : public PyBaseObject
{
	friend class ::Heap;

	PyObject *m_mapping{ nullptr };

  private:
	PyMappingProxy(PyType *);

	PyMappingProxy(PyObject *mapping);

  public:
	static PyResult<PyObject *> create(PyObject *mapping);
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __getitem__(PyObject *index);

	PyResult<PyObject *> get(PyTuple *, PyDict *) const;
	PyResult<PyObject *> items() const;
	PyResult<PyObject *> keys() const;
	PyResult<PyObject *> values() const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};
}// namespace py
