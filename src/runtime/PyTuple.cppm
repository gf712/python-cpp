module;

#include "memory/allocate.hpp"

export module py.runtime:tuple;
import :object;
import std;
import py.memory;

export namespace py {
class PyType;

class PyTupleIterator;

class PyTuple
	: public PyBaseObject
	, PySequence
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend PyTupleIterator;

	const std::vector<Value> m_elements;

  protected:
	PyTuple(PyType *);

	PyTuple();
	PyTuple(std::vector<Value> &&elements);
	PyTuple(PyType *, const std::vector<Value> elements);
	PyTuple(const std::vector<PyObject *> &elements);
	PyTuple(PyType *, const std::vector<PyObject *> &elements);

	void visit_graph(Visitor &) override;

  public:
	static PyResult<PyTuple *> create();
	static PyResult<PyTuple *> create(std::vector<Value> &&elements);
	static PyResult<PyTuple *> create(PyType *type, std::vector<Value> elements);
	static PyResult<PyTuple *> create(std::vector<PyObject *> &&elements);
	static PyResult<PyTuple *> create(const std::vector<PyObject *> &elements);
	static PyResult<PyTuple *> create(PyType *type, const std::vector<PyObject *> &elements);

	template<typename... Args> static PyResult<PyTuple *> create(Args &&...args)
	{
		return PyTuple::create(std::vector<Value>{ std::forward<Args>(args)... });
	}

	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<std::size_t> __len__() const;
	PyResult<PyObject *> __add__(const PyObject *other) const;
	PyResult<PyObject *> __eq__(const PyObject *other) const;
	PyResult<PyObject *> __getitem__(PyObject *key);

	PyResult<PyObject *> __getitem__(std::int64_t index);

	PyTupleIterator begin() const;
	PyTupleIterator end() const;

	// std::shared_ptr<PyTupleIterator> cbegin() const;
	// std::shared_ptr<PyTupleIterator> cend() const;

	const std::vector<Value> &elements() const { return m_elements; }
	std::size_t size() const { return m_elements.size(); }
	PyResult<PyObject *> operator[](std::size_t idx) const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

template<> PyTuple *as(PyObject *obj);
template<> const PyTuple *as(const PyObject *obj);

class PyTupleIterator : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend PyTuple;

	const PyTuple &m_pytuple;
	std::size_t m_current_index{ 0 };

	PyTupleIterator(const PyTuple &pytuple);
	PyTupleIterator(const PyTuple &pytuple, std::size_t position);

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
