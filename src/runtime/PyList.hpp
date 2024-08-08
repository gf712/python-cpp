#pragma once

#include "PyObject.hpp"

namespace py {

class PyList
	: public PyBaseObject
	, PySequence
{
	friend class ::Heap;

	std::vector<Value> m_elements;

	PyList(PyType *);

  public:
	static PyResult<PyList *> create(std::vector<Value> elements);
	static PyResult<PyList *> create(std::span<const Value> elements);
	static PyResult<PyList *> create();

	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __getitem__(PyObject *index);
	PyResult<std::monostate> __setitem__(PyObject *index, PyObject *value);
	PyResult<size_t> __len__() const;

	PyResult<PyObject *> __getitem__(int64_t index);
	PyResult<std::monostate> __setitem__(int64_t index, PyObject *value);
	PyResult<std::monostate> __delitem__(PyObject *key);

	PyResult<PyObject *> __add__(const PyObject *other) const;
	PyResult<PyObject *> __mul__(size_t count) const;
	PyResult<PyObject *> __eq__(const PyObject *other) const;

	const std::vector<Value> &elements() const { return m_elements; }
	std::vector<Value> &elements() { return m_elements; }

	void visit_graph(Visitor &) override;

	PyResult<PyObject *> append(PyObject *element);
	PyResult<PyObject *> extend(PyObject *iterable);
	PyResult<PyObject *> pop(PyObject *index);

	PyResult<PyObject *> __class_getitem__(PyType *cls, PyObject *args);
	PyResult<PyObject *> __reversed__() const;

	PyResult<PyObject *> sort();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

  private:
	PyList();
	PyList(std::vector<Value> elements);
};


class PyListIterator : public PyBaseObject
{
	friend class ::Heap;

	const PyList &m_pylist;
	size_t m_current_index{ 0 };

  public:
	PyListIterator(const PyList &pylist);

	std::string to_string() const override;

	void visit_graph(Visitor &) override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

class PyListReverseIterator : public PyBaseObject
{
	friend class ::Heap;

	std::optional<std::reference_wrapper<PyList>> m_pylist;
	size_t m_current_index{ 0 };

  private:
	PyListReverseIterator(PyType *);

	PyListReverseIterator(PyList &pylist, size_t start_index);

  public:
	static PyResult<PyListReverseIterator *> create(PyList &);

	void visit_graph(Visitor &) override;

	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __next__();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
