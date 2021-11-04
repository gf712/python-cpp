#pragma once

#include "PyObject.hpp"


class PyTupleIterator;

class PyTuple : public PyObject
{
	friend class Heap;

	std::vector<Value> m_elements;

  protected:
	PyTuple() : PyObject(PyObjectType::PY_TUPLE) {}

	PyTuple(std::vector<Value> &&elements)
		: PyObject(PyObjectType::PY_TUPLE), m_elements(std::move(elements))
	{}

	PyTuple(const std::vector<PyObject *> &elements) : PyObject(PyObjectType::PY_TUPLE)
	{
		m_elements.reserve(elements.size());
		for (auto *el : elements) { m_elements.push_back(el); }
	}

	void visit_graph(Visitor &) override;

  public:
	static PyTuple *create();

	static PyTuple *create(std::vector<Value> elements);

	static PyTuple *create(const std::vector<PyObject *> &elements);

	std::string to_string() const override;

	PyObject *repr_impl(Interpreter &interpreter) const override;
	PyObject *iter_impl(Interpreter &interpreter) const override;

	PyTupleIterator begin() const;
	PyTupleIterator end() const;

	// std::shared_ptr<PyTupleIterator> cbegin() const;
	// std::shared_ptr<PyTupleIterator> cend() const;

	const std::vector<Value> &elements() const { return m_elements; }
	size_t size() const { return m_elements.size(); }
	PyObject *operator[](size_t idx) const;
};


class PyTupleIterator : public PyObject
{
	friend class Heap;
	friend PyTuple;

	const PyTuple &m_pytuple;
	size_t m_current_index{ 0 };

  public:
	using difference_type = std::vector<Value>::difference_type;
	using value_type = PyObject *;
	using pointer = value_type *;
	using reference = value_type &;
	using iterator_category = std::forward_iterator_tag;

	PyTupleIterator(const PyTuple &pytuple)
		: PyObject(PyObjectType::PY_TUPLE_ITERATOR), m_pytuple(pytuple)
	{}

	PyTupleIterator(const PyTuple &pytuple, size_t position) : PyTupleIterator(pytuple)
	{
		m_current_index = position;
	}

	std::string to_string() const override;

	PyObject *repr_impl(Interpreter &interpreter) const override;
	PyObject *next_impl(Interpreter &interpreter) override;

	bool operator==(const PyTupleIterator &) const;
	PyObject *operator*() const;
	PyTupleIterator &operator++();
	PyTupleIterator &operator--();
};


template<> inline PyTuple *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_TUPLE) { return static_cast<PyTuple *>(node); }
	return nullptr;
}


template<> inline const PyTuple *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_TUPLE) { return static_cast<const PyTuple *>(node); }
	return nullptr;
}
