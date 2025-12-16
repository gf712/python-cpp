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

	PySet(PyType *);

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
	PyResult<PyObject *> __le__(const PyObject *other) const;
	PyResult<PyObject *> __lt__(const PyObject *other) const;
	PyResult<bool> __contains__(const PyObject *value) const;

	PyResult<PyObject *> __and__(PyObject *other);
	PyResult<PyObject *> __or__(PyObject *obj);

	const SetType &elements() const { return m_elements; }
	SetType &elements() { return m_elements; }

	void visit_graph(Visitor &) override;

	PyResult<PyObject *> add(PyObject *element);
	PyResult<PyObject *> discard(PyObject *element);
	PyResult<PyObject *> remove(PyObject *element);
	PyResult<PySet *> update(PyObject *iterable);
	PyResult<PySet *> intersection(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> pop();
	PyResult<PyObject *> issubset(const PyObject *other) const;
	PyResult<PyObject *> union_(PyTuple *others, PyDict *kwargs) const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

  private:
	PyResult<std::monostate>
		union_helper(const PyObject *other, SetType &result, bool strict) const;

	PySet();
	PySet(SetType elements);
};


class PySetIterator : public PyBaseObject
{
	friend class ::Heap;

	const std::variant<std::monostate,
		std::reference_wrapper<const PySet>,
		std::reference_wrapper<const PyFrozenSet>>
		m_pyset;
	size_t m_current_index{ 0 };

	PySetIterator(PyType *);

  public:
	PySetIterator(const PySet &pyset);
	PySetIterator(const PyFrozenSet &pyset);

	std::string to_string() const override;

	void visit_graph(Visitor &) override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
