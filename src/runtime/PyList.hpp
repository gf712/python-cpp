#pragma once

#include "PyObject.hpp"

class PyList : public PyBaseObject
{
	friend class Heap;

	std::vector<Value> m_elements;

  public:
	static PyList *create(std::vector<Value> elements);
	static PyList *create();

	std::string to_string() const override;

	PyObject *__repr__() const;
	PyObject *__iter__() const;
	PyObject *__len__() const;

	const std::vector<Value> &elements() const { return m_elements; }
	std::vector<Value> &elements() { return m_elements; }

	void visit_graph(Visitor &) override;

	PyObject *append(PyTuple *args, PyDict *kwargs);

	void sort();

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;

  private:
	PyList();
	PyList(std::vector<Value> elements);
};


class PyListIterator : public PyBaseObject
{
	friend class Heap;

	const PyList &m_pylist;
	size_t m_current_index{ 0 };

  public:
	PyListIterator(const PyList &pylist);

	std::string to_string() const override;

	void visit_graph(Visitor &) override;

	PyObject *__repr__() const;
	PyObject *__next__();

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};
