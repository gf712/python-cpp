#pragma once

#include "PyObject.hpp"
#include <unordered_set>

namespace py {

class PySet : public PyBaseObject
{
	friend class ::Heap;

  public:
	using SetType = std::unordered_set<Value, ValueHash>;

  private:
	SetType m_elements;

  public:
	static PyResult<PySet *> create(SetType elements);
	static PyResult<PySet *> create();

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

	PyResult<PyObject *> add(PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;

  private:
	PySet();
	PySet(SetType elements);
};


class PySetIterator : public PyBaseObject
{
	friend class ::Heap;

	const PySet &m_pyset;
	size_t m_current_index{ 0 };

  public:
	PySetIterator(const PySet &pyset);

	std::string to_string() const override;

	void visit_graph(Visitor &) override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};

}// namespace py