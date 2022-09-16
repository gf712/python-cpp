#pragma once

#include "PyObject.hpp"

namespace py {

class PyList : public PyBaseObject
{
	friend class ::Heap;

	std::vector<Value> m_elements;

  public:
	static PyResult<PyList *> create(std::vector<Value> elements);
	static PyResult<PyList *> create();

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __getitem__(PyObject *index);
	PyResult<size_t> __len__() const;
	PyResult<PyObject *> __eq__(const PyObject *other) const;

	const std::vector<Value> &elements() const { return m_elements; }
	std::vector<Value> &elements() { return m_elements; }

	void visit_graph(Visitor &) override;

	PyResult<PyObject *> append(PyObject *element);
	PyResult<PyObject *> extend(PyObject *iterable);

	PyResult<PyObject *> __class_getitem__(PyType *cls, PyObject *args);

	void sort();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;

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
	PyType *type() const override;
};

}// namespace py