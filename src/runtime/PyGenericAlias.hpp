#pragma once

#include "PyObject.hpp"

namespace py {
class PyGenericAlias : public PyBaseObject
{
	friend ::Heap;

	PyObject *m_origin{ nullptr };
	PyTuple *m_args{ nullptr };
	PyObject *m_parameters{ nullptr };

	PyGenericAlias(PyObject *origin, PyTuple *args, PyObject *parameters);

  public:
	static PyResult<PyGenericAlias *>
		create(PyObject *origin, PyObject *args, PyObject *parameters);

	static PyResult<PyGenericAlias *> create(PyObject *origin, PyObject *args);

	PyResult<PyObject *> __repr__() const;
	std::string to_string() const override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
	void visit_graph(Visitor &visitor) override;
};
}// namespace py