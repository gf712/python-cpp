#pragma once

#include "PyObject.hpp"
#include <unordered_set>

namespace py {

class PyFrozenSet : public PyBaseObject
{
	friend class ::Heap;

  public:
	using SetType = std::unordered_set<Value, ValueHash>;

  private:
	SetType m_elements;

	PyFrozenSet(PyType *);

  public:
	static PyResult<PyFrozenSet *> create(SetType elements);
	static PyResult<PyFrozenSet *> create();

	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<size_t> __len__() const;
	PyResult<PyObject *> __eq__(const PyObject *other) const;
	PyResult<bool> __contains__(const PyObject *value) const;


	const SetType &elements() const { return m_elements; }
	SetType &elements() { return m_elements; }

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

  private:
	PyFrozenSet();
	PyFrozenSet(SetType elements);
};

}// namespace py
