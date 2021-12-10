#pragma once

#include "PyObject.hpp"


class PyTupleIterator;

class PyTuple : public PyBaseObject
{
	friend class Heap;

	std::vector<Value> m_elements;

  protected:
	PyTuple();
	PyTuple(std::vector<Value> &&elements);
	PyTuple(const std::vector<PyObject *> &elements);

	void visit_graph(Visitor &) override;

  public:
	static PyTuple *create();
	static PyTuple *create(std::vector<Value> elements);
	static PyTuple *create(const std::vector<PyObject *> &elements);
	template<typename... Args> static PyTuple *create(Args &&... args)
	{
		return PyTuple::create(std::vector<Value>{ std::forward<Args>(args)... });
	}

	std::string to_string() const override;

	PyObject *__repr__() const;
	PyObject *__iter__() const;
	PyObject *__len__() const;

	PyTupleIterator begin() const;
	PyTupleIterator end() const;

	// std::shared_ptr<PyTupleIterator> cbegin() const;
	// std::shared_ptr<PyTupleIterator> cend() const;

	const std::vector<Value> &elements() const { return m_elements; }
	size_t size() const { return m_elements.size(); }
	PyObject *operator[](size_t idx) const;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};


class PyTupleIterator : public PyBaseObject
{
	friend class Heap;
	friend PyTuple;

	const PyTuple &m_pytuple;
	size_t m_current_index{ 0 };

	PyTupleIterator(const PyTuple &pytuple);
	PyTupleIterator(const PyTuple &pytuple, size_t position);

  public:
	using difference_type = std::vector<Value>::difference_type;
	using value_type = PyObject *;
	using pointer = value_type *;
	using reference = value_type &;
	using iterator_category = std::forward_iterator_tag;

	std::string to_string() const override;

	PyObject *__repr__() const;
	PyObject *__next__();

	bool operator==(const PyTupleIterator &) const;
	PyObject *operator*() const;
	PyTupleIterator &operator++();
	PyTupleIterator &operator--();

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};
