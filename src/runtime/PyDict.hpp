#pragma once

#include "PyObject.hpp"
#include "runtime/Value.hpp"

#include <tsl/ordered_map.h>

#include <variant>

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
	using MapType = tsl::ordered_map<Value, Value, ValueHash>;

  private:
	friend class ::Heap;
	friend PyDictItems;
	friend PyDictItemsIterator;

	MapType m_map;

	PyDict(MapType &&map);
	PyDict(const MapType &map);
	PyDict();
	PyDict(PyType *);

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
	PyResult<size_t> __len__() const;

	const MapType &map() const { return m_map; }

	void insert(const Value &key, const Value &value);
	void remove(const Value &key);
	PyResult<PyObject *> merge(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> update(PyObject *other);
	PyResult<PyObject *> update(PyDict *other);
	PyResult<PyDictKeys *> keys() const;
	PyResult<PyDictValues *> values() const;
	PyResult<PyObject *> setdefault(PyTuple *args, PyDict *kwargs);

	static PyResult<PyObject *> fromkeys(PyObject *iterable, PyObject *value);

	std::optional<Value> operator[](Value key) const;

	PyResult<PyObject *> get(PyObject *, PyObject *) const;
	PyResult<PyObject *> pop(PyObject *, PyObject *);

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

  private:
	PyResult<std::monostate> merge(PyObject *other, bool override);
	PyResult<std::monostate> merge(PyDict *other, bool override);
	PyResult<std::monostate> merge_from_seq_2(PyObject *other, bool override);
};

class PyDictItems : public PyBaseObject
{
	friend class ::Heap;
	friend PyDict;
	friend PyDictItemsIterator;

	const std::optional<std::reference_wrapper<const PyDict>> m_pydict;

	PyDictItems(PyType *);

  public:
	static PyResult<PyDictItems *> create(const PyDict &pydict);

	PyDictItems(const PyDict &pydict);

	PyDictItemsIterator begin() const;
	PyDictItemsIterator end() const;

	PyResult<PyObject *> __iter__() const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};


class PyDictKeys : public PyBaseObject
{
	friend class ::Heap;
	friend PyDict;
	friend PyDictKeyIterator;

	const std::optional<std::reference_wrapper<const PyDict>> m_pydict;

	PyDictKeys(PyType *);

	PyDictKeys(const PyDict &pydict);

  public:
	static PyResult<PyDictKeys *> create(const PyDict &pydict);

	PyDictKeyIterator begin() const;
	PyDictKeyIterator end() const;

	PyResult<PyObject *> __iter__() const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};


class PyDictValues : public PyBaseObject
{
	friend class ::Heap;
	friend PyDict;
	friend PyDictValueIterator;

	const std::optional<std::reference_wrapper<const PyDict>> m_pydict;

	PyDictValues(PyType *);

	PyDictValues(const PyDict &pydict);

  public:
	static PyResult<PyDictValues *> create(const PyDict &pydict);

	PyDictValueIterator begin() const;
	PyDictValueIterator end() const;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

class PyDictItemsIterator : public PyBaseObject
{
	friend class ::Heap;
	friend PyDictItems;

	const std::optional<std::reference_wrapper<const PyDictItems>> m_pydictitems;
	PyDict::MapType::const_iterator m_current_iterator;

	PyDictItemsIterator(PyType *);

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
	PyType *static_type() const override;
};


class PyDictKeyIterator : public PyBaseObject
{
	friend class ::Heap;
	friend PyDictItems;
	friend PyDictKeys;

	const std::optional<std::reference_wrapper<const PyDictKeys>> m_pydictkeys;
	PyDict::MapType::const_iterator m_current_iterator;

	PyDictKeyIterator(PyType *);

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
	PyType *static_type() const override;
};

class PyDictValueIterator : public PyBaseObject
{
	friend class ::Heap;
	friend PyDictItems;
	friend PyDictValues;

	const std::optional<std::reference_wrapper<const PyDictValues>> m_pydictvalues;
	PyDict::MapType::const_iterator m_current_iterator;

	PyDictValueIterator(PyType *);

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
	PyType *static_type() const override;
};

}// namespace py
