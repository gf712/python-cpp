#pragma once

#include "PyObject.hpp"

class PyList : public PyObject
{
	friend class Heap;

	std::vector<Value> m_elements;

  public:
	static PyList *create(std::vector<Value> elements);
	static PyList *create();

	std::string to_string() const override;

	PyObject *repr_impl(Interpreter &interpreter) const override;
	PyObject *iter_impl(Interpreter &interpreter) const override;

	const std::vector<Value> &elements() const { return m_elements; }

	void visit_graph(Visitor &) override;

	PyObject *append(PyObject *);

	void sort();

  private:
	PyList();
	PyList(std::vector<Value> elements);
};


class PyListIterator : public PyObject
{
	friend class Heap;

	const PyList &m_pylist;
	size_t m_current_index{ 0 };

  public:
	PyListIterator(const PyList &pylist)
		: PyObject(PyObjectType::PY_LIST_ITERATOR), m_pylist(pylist)
	{}

	std::string to_string() const override;

	void visit_graph(Visitor &) override;

	PyObject *repr_impl(Interpreter &interpreter) const override;
	PyObject *next_impl(Interpreter &interpreter) override;
};

template<> inline PyList *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_LIST) { return static_cast<PyList *>(node); }
	return nullptr;
}

template<> inline const PyList *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_LIST) { return static_cast<const PyList *>(node); }
	return nullptr;
}