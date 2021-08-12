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

	std::shared_ptr<PyDictItems> items() const;

	size_t size() const { return m_map.size(); }

	std::string to_string() const override;
	std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const override;

	const MapType &map() const { return m_map; }

	void insert(const Value &key, const Value &value);
	Value operator[](Value key) const;
};

class PyDictItems : public PyObject
{
	friend class Heap;
	friend PyDict;
	friend PyDictItemsIterator;

	std::shared_ptr<const PyDict> m_pydict;

  public:
	PyDictItems(std::shared_ptr<const PyDict> pydict)
		: PyObject(PyObjectType::PY_DICT_ITEMS), m_pydict(std::move(pydict))
	{}

	PyDictItemsIterator begin() const;
	PyDictItemsIterator end() const;

	std::string to_string() const override;
};


class PyDictItemsIterator : public PyObject
{
	friend class Heap;
	friend PyDictItems;

	std::shared_ptr<const PyDictItems> m_pydictitems;
	PyDict::MapType::const_iterator m_current_iterator;

  public:
	using difference_type = PyDict::MapType::difference_type;
	using value_type = std::shared_ptr<PyTuple>;
	using pointer = value_type *;
	using reference = value_type &;
	using iterator_category = std::forward_iterator_tag;

	PyDictItemsIterator(std::shared_ptr<const PyDictItems> pydict)
		: PyObject(PyObjectType::PY_DICT_ITEMS_ITERATOR), m_pydictitems(std::move(pydict)),
		  m_current_iterator(m_pydictitems->m_pydict->map().begin())
	{}

	PyDictItemsIterator(std::shared_ptr<const PyDictItems> pydict, size_t position)
		: PyDictItemsIterator(pydict)
	{
		std::advance(m_current_iterator, position);
	}

	std::string to_string() const override;

	std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> next_impl(Interpreter &interpreter) override;

	bool operator==(const PyDictItemsIterator &) const;
	value_type operator*() const;
	PyDictItemsIterator &operator++();
};


template<> inline std::shared_ptr<PyDict> as(std::shared_ptr<PyObject> node)
{
	if (node->type() == PyObjectType::PY_DICT) { return std::static_pointer_cast<PyDict>(node); }
	return nullptr;
}