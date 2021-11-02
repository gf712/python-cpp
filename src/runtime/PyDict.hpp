#pragma once

#include "PyObject.hpp"

class PyDictItems;
class PyDictItemsIterator;

class PyDict : public PyObject
{
  public:
	using MapType = std::unordered_map<Value, Value, ValueHash, ValueEqual>;

  private:
	friend class Heap;
	friend PyDictItems;
	friend PyDictItemsIterator;

	MapType m_map;

  public:
	PyDict(MapType &&map) : PyObject(PyObjectType::PY_DICT), m_map(std::move(map)) {}
	PyDict(const MapType &map) : PyObject(PyObjectType::PY_DICT), m_map(map) {}
	PyDict() : PyObject(PyObjectType::PY_DICT) {}

	PyDictItems *items() const;

	size_t size() const { return m_map.size(); }

	std::string to_string() const override;
	PyObject *repr_impl(Interpreter &interpreter) const override;

	const MapType &map() const { return m_map; }

	void insert(const Value &key, const Value &value);
	Value operator[](Value key) const;

	void visit_graph(Visitor &) override;
};

class PyDictItems : public PyObject
{
	friend class Heap;
	friend PyDict;
	friend PyDictItemsIterator;

	const PyDict &m_pydict;

  public:
	PyDictItems(const PyDict &pydict) : PyObject(PyObjectType::PY_DICT_ITEMS), m_pydict(pydict) {}

	PyDictItemsIterator begin() const;
	PyDictItemsIterator end() const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;
};


class PyDictItemsIterator : public PyObject
{
	friend class Heap;
	friend PyDictItems;

	const PyDictItems &m_pydictitems;
	PyDict::MapType::const_iterator m_current_iterator;

  public:
	using difference_type = PyDict::MapType::difference_type;
	using value_type = PyTuple *;
	using pointer = value_type *;
	using reference = value_type &;
	using iterator_category = std::forward_iterator_tag;

	PyDictItemsIterator(const PyDictItems &pydict)
		: PyObject(PyObjectType::PY_DICT_ITEMS_ITERATOR), m_pydictitems(pydict),
		  m_current_iterator(m_pydictitems.m_pydict.map().begin())
	{}

	PyDictItemsIterator(const PyDictItems &pydict, size_t position) : PyDictItemsIterator(pydict)
	{
		std::advance(m_current_iterator, position);
	}

	std::string to_string() const override;

	PyObject *repr_impl(Interpreter &interpreter) const override;
	PyObject *next_impl(Interpreter &interpreter) override;

	bool operator==(const PyDictItemsIterator &) const;
	value_type operator*() const;
	PyDictItemsIterator &operator++();
	
	void visit_graph(Visitor &) override;
};


template<> inline PyDict *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_DICT) { return static_cast<PyDict *>(node); }
	return nullptr;
}

template<> inline const PyDict *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_DICT) { return static_cast<const PyDict *>(node); }
	return nullptr;
}