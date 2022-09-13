#pragma once

#include "PyObject.hpp"

namespace py {

class PyDictItems;
class PyDictItemsIterator;

class PyDictKeys;
class PyDictKeyIterator;

class PyDictValues;
class PyDictValueIterator;

class PyDict : public PyBaseObject
{
  public:
	using MapType = std::unordered_map<Value, Value, ValueHash>;

  private:
	friend class ::Heap;
	friend PyDictItems;
	friend PyDictItemsIterator;

	MapType m_map;

	PyDict(MapType &&map);
	PyDict(const MapType &map);
	PyDict();

  public:
	static PyResult<PyDict *> create();
	static PyResult<PyDict *> create(MapType &&map);
	static PyResult<PyDict *> create(const MapType &map);

	PyDictItems *items() const;

	size_t size() const { return m_map.size(); }

	std::string to_string() const override;
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __getitem__(PyObject *key);
	PyResult<std::monostate> __setitem__(PyObject *key, PyObject *value);
	PyResult<std::monostate> __delitem__(PyObject *key);
	PyResult<PyObject *> __eq__(const PyObject *other) const;

	const MapType &map() const { return m_map; }

	void insert(const Value &key, const Value &value);
	void remove(const Value &key);
	PyResult<PyObject *> merge(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> update(PyDict *other);
	static PyResult<PyObject *> fromkeys(PyObject *iterable, PyObject *value);

	Value operator[](Value key) const;

	PyResult<PyObject *> get(PyObject *, PyObject *) const;
	PyResult<PyObject *> pop(PyObject *, PyObject *);

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};

class PyDictItems : public PyBaseObject
{
	friend class ::Heap;
	friend PyDict;
	friend PyDictItemsIterator;

	const PyDict &m_pydict;

  public:
	static PyResult<PyDictItems *> create(const PyDict &pydict);

	PyDictItems(const PyDict &pydict);

	PyDictItemsIterator begin() const;
	PyDictItemsIterator end() const;

	PyResult<PyObject *> __iter__() const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};


class PyDictKeys : public PyBaseObject
{
	friend class ::Heap;
	friend PyDict;
	friend PyDictKeyIterator;

	const PyDict &m_pydict;

	PyDictKeys(const PyDict &pydict);

  public:
	static PyResult<PyDictKeys *> create(const PyDict &pydict);

	PyDictKeyIterator begin() const;
	PyDictKeyIterator end() const;

	PyResult<PyObject *> __iter__() const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};


class PyDictValues : public PyBaseObject
{
	friend class ::Heap;
	friend PyDict;
	friend PyDictValueIterator;

	const PyDict &m_pydict;

	PyDictValues(const PyDict &pydict);

  public:
	static PyResult<PyDictValues *> create(const PyDict &pydict);

	PyDictValueIterator begin() const;
	PyDictValueIterator end() const;

	PyResult<PyObject *> __iter__() const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};

class PyDictItemsIterator : public PyBaseObject
{
	friend class ::Heap;
	friend PyDictItems;

	const PyDictItems &m_pydictitems;
	PyDict::MapType::const_iterator m_current_iterator;

	PyDictItemsIterator(const PyDictItems &pydict_items);
	PyDictItemsIterator(const PyDictItems &pydict_items, size_t position);

  public:
	using difference_type = PyDict::MapType::difference_type;
	using value_type = PyTuple *;
	using pointer = value_type *;
	using reference = value_type &;
	using iterator_category = std::forward_iterator_tag;

	static PyResult<PyDictItemsIterator *> create(const PyDictItems &pydict_items);
	static PyResult<PyDictItemsIterator *> create(const PyDictItems &pydict_items, size_t position);

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();

	bool operator==(const PyDictItemsIterator &) const;
	value_type operator*() const;
	PyDictItemsIterator &operator++();

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};


class PyDictKeyIterator : public PyBaseObject
{
	friend class ::Heap;
	friend PyDictItems;
	friend PyDictKeys;

	const PyDictKeys &m_pydictkeys;
	PyDict::MapType::const_iterator m_current_iterator;

	PyDictKeyIterator(const PyDictKeys &pydict_keys);
	PyDictKeyIterator(const PyDictKeys &pydict_keys, size_t position);

  public:
	using difference_type = PyDict::MapType::difference_type;
	using value_type = PyObject *;
	using pointer = value_type *;
	using reference = value_type &;
	using iterator_category = std::forward_iterator_tag;

	static PyResult<PyDictKeyIterator *> create(const PyDictKeys &pydict_keys);
	static PyResult<PyDictKeyIterator *> create(const PyDictKeys &pydict_keys, size_t position);

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();

	bool operator==(const PyDictKeyIterator &) const;
	value_type operator*() const;
	PyDictKeyIterator &operator++();

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};

class PyDictValueIterator : public PyBaseObject
{
	friend class ::Heap;
	friend PyDictItems;
	friend PyDictValues;

	const PyDictValues &m_pydictvalues;
	PyDict::MapType::const_iterator m_current_iterator;

	PyDictValueIterator(const PyDictValues &pydict_values);
	PyDictValueIterator(const PyDictValues &pydict_values, size_t position);

  public:
	using difference_type = PyDict::MapType::difference_type;
	using value_type = PyObject *;
	using pointer = value_type *;
	using reference = value_type &;
	using iterator_category = std::forward_iterator_tag;

	static PyResult<PyDictValueIterator *> create(const PyDictValues &pydict_values);
	static PyResult<PyDictValueIterator *> create(const PyDictValues &pydict_values,
		size_t position);

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();

	bool operator==(const PyDictValueIterator &) const;
	value_type operator*() const;
	PyDictValueIterator &operator++();

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};

}// namespace py