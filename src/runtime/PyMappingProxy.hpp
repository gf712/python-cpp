#pragma once

#include "PyObject.hpp"

namespace py {
class PyMappingProxy : public PyBaseObject
{
	friend class ::Heap;

	PyObject *m_mapping{ nullptr };

  private:
	PyMappingProxy(PyObject *mapping);

  public:
	static PyResult<PyObject *> create(PyObject *mapping);
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __getitem__(PyObject *index);

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};
}// namespace py
