#pragma once

#include "PyObject.hpp"

namespace py {

class PyTuple
	: public PyBaseObject
	, PySequence
{
	friend class ::Heap;
	friend class PyTupleIterator;

	const std::vector<Value> m_elements;

  protected:
	PyTuple(PyType *);

	PyTuple();
	PyTuple(std::vector<Value> &&elements);
	PyTuple(const std::vector<PyObject *> &elements);

	void visit_graph(Visitor &) override;

  public:
	static PyResult<PyTuple *> create();
	static PyResult<PyTuple *> create(std::vector<Value> &&elements);
	static PyResult<PyTuple *> create(std::vector<PyObject *> &&elements);
	static PyResult<PyTuple *> create(const std::vector<PyObject *> &elements);
	template<typename... Args> static PyResult<PyTuple *> create(Args &&...args)
	{
		return PyTuple::create(std::vector<Value>{ std::forward<Args>(args)... });
	}

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<size_t> __len__() const;
	PyResult<PyObject *> __add__(const PyObject *other) const;
	PyResult<PyObject *> __eq__(const PyObject *other) const;
	PyResult<PyObject *> __getitem__(PyObject *key);

	PyResult<PyObject *> __getitem__(int64_t index);

	PyTupleIterator begin() const;
	PyTupleIterator end() const;

	// std::shared_ptr<PyTupleIterator> cbegin() const;
	// std::shared_ptr<PyTupleIterator> cend() const;

	const std::vector<Value> &elements() const { return m_elements; }
	size_t size() const { return m_elements.size(); }
	PyResult<PyObject *> operator[](size_t idx) const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

template<> PyTuple *as(PyObject *obj);
template<> const PyTuple *as(const PyObject *obj);

class PyTupleIterator : public PyBaseObject
{
	friend class ::Heap;
	friend PyTuple;

	const PyTuple &m_pytuple;
	size_t m_current_index{ 0 };

	PyTupleIterator(const PyTuple &pytuple);
	PyTupleIterator(const PyTuple &pytuple, size_t position);

  protected:
	void visit_graph(Visitor &) override;

  public:
	using difference_type = std::vector<Value>::difference_type;
	using value_type = PyObject *;
	using pointer = value_type *;
	using reference = value_type &;
	using iterator_category = std::forward_iterator_tag;

	static PyResult<PyTupleIterator *> create(const PyTuple &pytuple);

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();

	bool operator==(const PyTupleIterator &) const;
	PyResult<PyObject *> operator*() const;
	PyTupleIterator &operator++();
	PyTupleIterator &operator--();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
