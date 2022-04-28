#pragma once

#include "PyObject.hpp"

namespace py {

class PyDictItems;
class PyDictItemsIterator;

class PyDict : public PyBaseObject
{
  public:
	using MapType = std::unordered_map<Value, Value, ValueHash, ValueEqual>;

  private:
	friend class ::Heap;
	friend PyDictItems;
	friend PyDictItemsIterator;

	MapType m_map;

	PyDict(MapType &&map);
	PyDict(const MapType &map);
	PyDict();

  public:
	static PyResult create();
	static PyResult create(MapType &&map);
	static PyResult create(const MapType &map);

	PyDictItems *items() const;

	size_t size() const { return m_map.size(); }

	std::string to_string() const override;
	PyResult __repr__() const;
	PyResult __eq__(const PyObject *other) const;

	const MapType &map() const { return m_map; }

	void insert(const Value &key, const Value &value);
	PyResult merge(PyTuple *args, PyDict *kwargs);
	PyResult update(PyDict *other);

	Value operator[](Value key) const;

	PyResult get(PyObject *, PyObject *) const;

	void visit_graph(Visitor &) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

class PyDictItems : public PyBaseObject
{
	friend class ::Heap;
	friend PyDict;
	friend PyDictItemsIterator;

	const PyDict &m_pydict;

  public:
	PyDictItems(const PyDict &pydict);

	PyDictItemsIterator begin() const;
	PyDictItemsIterator end() const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};


class PyDictItemsIterator : public PyBaseObject
{
	friend class ::Heap;
	friend PyDictItems;

	const PyDictItems &m_pydictitems;
	PyDict::MapType::const_iterator m_current_iterator;

  public:
	using difference_type = PyDict::MapType::difference_type;
	using value_type = PyTuple *;
	using pointer = value_type *;
	using reference = value_type &;
	using iterator_category = std::forward_iterator_tag;

	PyDictItemsIterator(const PyDictItems &pydict);

	PyDictItemsIterator(const PyDictItems &pydict, size_t position);

	std::string to_string() const override;

	PyResult __repr__() const;
	PyResult __next__();

	bool operator==(const PyDictItemsIterator &) const;
	value_type operator*() const;
	PyDictItemsIterator &operator++();

	void visit_graph(Visitor &) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py