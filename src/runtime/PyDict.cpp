#include "PyDict.hpp"
#include "KeyError.hpp"
#include "MemoryError.hpp"
#include "PyBool.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "StopIteration.hpp"
#include "ValueError.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

#include <unordered_set>
#include <variant>

namespace py {

static std::unordered_set<PyObject *> visited_dict_values;

template<> PyDict *as(PyObject *obj)
{
	if (obj->type() == types::dict()) { return static_cast<PyDict *>(obj); }
	return nullptr;
}

template<> const PyDict *as(const PyObject *obj)
{
	if (obj->type() == types::dict()) { return static_cast<const PyDict *>(obj); }
	return nullptr;
}

PyDict::PyDict(MapType &&map)
	: PyBaseObject(types::BuiltinTypes::the().dict()), m_map(std::move(map))
{}

PyDict::PyDict(const MapType &map) : PyBaseObject(types::BuiltinTypes::the().dict()), m_map(map) {}

PyDict::PyDict() : PyBaseObject(types::BuiltinTypes::the().dict()) {}

PyDict::PyDict(PyType *type) : PyBaseObject(type) {}

PyResult<PyDict *> PyDict::create()
{
	auto *result = VirtualMachine::the().heap().allocate<PyDict>();
	if (!result) { return Err(memory_error(sizeof(PyDict))); }
	return Ok(result);
}

PyResult<PyDict *> PyDict::create(MapType &&map)
{
	auto *result = VirtualMachine::the().heap().allocate<PyDict>(std::move(map));
	if (!result) { return Err(memory_error(sizeof(PyDict))); }
	return Ok(result);
}

PyResult<PyDict *> PyDict::create(const MapType &map)
{
	auto *result = VirtualMachine::the().heap().allocate<PyDict>(map);
	if (!result) { return Err(memory_error(sizeof(PyDict))); }
	return Ok(result);
}

std::string PyDict::to_string() const
{
	if (m_map.empty()) { return "{}"; }

	std::ostringstream os;
	os << "{";

	auto it = m_map.begin();
	while (std::next(it) != m_map.end()) {
		std::visit(overloaded{ [&os](PyObject *key) {
								  auto r = key->repr();
								  ASSERT(r.is_ok())
								  os << r.unwrap()->to_string();
							  },
					   [&os](const auto &key) { os << key; } },
			it->first);
		os << ": ";
		std::visit(overloaded{ [&os, this](PyObject *value) {
								  if (value == this) {
									  os << "{...}";
								  } else {
									  auto r = value->repr();
									  ASSERT(r.is_ok())
									  os << r.unwrap()->to_string();
								  }
							  },
					   [&os](const auto &value) { os << value; } },
			it->second);
		os << ", ";

		std::advance(it, 1);
	}
	std::visit(overloaded{ [&os](PyObject *key) {
							  auto r = key->repr();
							  ASSERT(r.is_ok())
							  os << r.unwrap()->to_string() << ": ";
						  },
				   [&os](const auto &key) { os << key << ": "; } },
		it->first);
	std::visit(overloaded{ [&os, this](PyObject *value) {
							  if (value == this) {
								  os << "{...}";
							  } else {
								  auto r = value->repr();
								  ASSERT(r.is_ok())
								  os << r.unwrap()->to_string();
							  }
						  },
				   [&os](const auto &value) { os << value; } },
		it->second);
	os << "}";

	return os.str();
}

PyResult<PyObject *> PyDict::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyDict::__iter__() const
{
	return PyDictKeys::create(*this).and_then(
		[](PyDictKeys *keys) { return PyDictKeyIterator::create(*keys); });
}

PyResult<PyObject *> PyDict::__getitem__(PyObject *key)
{
	if (auto it = m_map.find(key); it != m_map.end()) { return PyObject::from(it->second); }
	if (type() != types::dict()) {
		auto missing = lookup_attribute(PyString::create("__missing__").unwrap());
		if (std::get<1>(missing) == LookupAttrResult::FOUND) {
			return std::get<0>(missing).and_then([key](auto *method) -> PyResult<PyObject *> {
				return method->call(PyTuple::create(key).unwrap(), nullptr);
			});
		}
		if (std::get<0>(missing).is_err()) { return std::get<0>(missing); }
	}
	return Err(key_error("{}", key->to_string()));
}

PyResult<std::monostate> PyDict::__setitem__(PyObject *key, PyObject *value)
{
	m_map.insert_or_assign(key, value);
	return Ok(std::monostate{});
}

PyResult<std::monostate> PyDict::__delitem__(PyObject *key)
{
	if (auto it = m_map.find(key); it != m_map.end()) {
		m_map.erase(it);
		return Ok(std::monostate{});
	}
	return Err(key_error("{}", key->to_string()));
}

PyResult<PyObject *> PyDict::__eq__(const PyObject *other) const
{
	if (!as<PyDict>(other)) { return Ok(py_false()); }

	return Ok(m_map == as<PyDict>(other)->map() ? py_true() : py_false());
}

PyResult<size_t> PyDict::__len__() const { return Ok(m_map.size()); }

PyDictItems *PyDict::items() const
{
	return VirtualMachine::the().heap().allocate<PyDictItems>(*this);
}

std::optional<Value> PyDict::operator[](Value key) const
{
	if (auto iter = m_map.find(key); iter != m_map.end()) {
		return iter->second;
	} else {
		return {};
	}
}

void PyDict::insert(const Value &key, const Value &value) { m_map.insert_or_assign(key, value); }

void PyDict::remove(const Value &key) { m_map.erase(key); }

PyResult<PyObject *> PyDict::merge(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyObject *, bool>::unpack_tuple(args,
		kwargs,
		"merge",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 2>{},
		false /* override */);

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [other_dict, override] = result.unwrap();
	return merge(other_dict, override).and_then([](auto) -> PyResult<PyObject *> {
		return Ok(py_none());
	});
}


void PyDict::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto &[key, value] : m_map) {
		if (std::holds_alternative<PyObject *>(value)) {
			if (std::get<PyObject *>(value) && std::get<PyObject *>(value) != this)
				visitor.visit(*std::get<PyObject *>(value));
		}
		if (std::holds_alternative<PyObject *>(key)) {
			if (std::get<PyObject *>(key) && std::get<PyObject *>(key) != this)
				visitor.visit(*std::get<PyObject *>(key));
		}
	}
}

PyResult<PyDictKeys *> PyDict::keys() const { return PyDictKeys::create(*this); }

PyResult<PyDictValues *> PyDict::values() const { return PyDictValues::create(*this); }

PyType *PyDict::static_type() const { return types::dict(); }

PyResult<PyObject *> PyDict::get(PyObject *key, PyObject *default_value) const
{
	if (auto it = m_map.find(key); it != m_map.end()) {
		return PyObject::from(it->second);
	} else if (default_value) {
		return Ok(default_value);
	}
	return Ok(py_none());
}

PyResult<PyObject *> PyDict::pop(PyObject *key, PyObject *default_value)
{
	if (auto it = m_map.find(key); it != m_map.end()) {
		auto result = it->second;
		m_map.erase(it);
		return PyObject::from(result);
	} else if (default_value) {
		return Ok(default_value);
	}
	return Err(key_error("{}", key->repr().unwrap()->to_string()));
}

PyResult<std::monostate> PyDict::merge(PyObject *other, bool override)
{
	if (other->type()->issubclass(types::dict())
		&& other->type()->underlying_type().__iter__.has_value()
		&& get_address(*other->type()->underlying_type().__iter__)
			   == get_address(*types::dict()->underlying_type().__iter__)) {
		return merge(static_cast<PyDict *>(other), override);
	}

	auto iter_ = other->get_attribute(PyString::create("keys").unwrap())
					 .and_then([](auto *keys_method) {
						 return keys_method->call(PyTuple::create().unwrap(), nullptr);
					 })
					 .and_then([](auto *keys) { return keys->iter(); });

	if (iter_.is_err()) { return Err(iter_.unwrap_err()); }
	auto *iter = iter_.unwrap();

	auto key_ = iter->next();
	if (key_.is_err()) { return Err(key_.unwrap_err()); }

	while (key_.is_ok()) {
		if (!override) {
			if (!m_map.contains(key_.unwrap())) {
				return KeyError::create(PyTuple::create(key_.unwrap()).unwrap())
					.and_then([](auto err) -> PyResult<std::monostate> { return Err(err); });
			}
		}

		auto value_ = other->getitem(key_.unwrap());
		if (value_.is_err()) { return Err(value_.unwrap_err()); }

		m_map.insert_or_assign(key_.unwrap(), value_.unwrap());
		key_ = iter->next();
	}

	if (!key_.unwrap_err()->type()->issubclass(types::stop_iteration())) {
		return Err(key_.unwrap_err());
	}

	return Ok(std::monostate{});
}


PyResult<std::monostate> PyDict::merge_from_seq_2(PyObject *other, bool override)
{
	auto iter = other->iter();
	if (iter.is_err()) { return Err(iter.unwrap_err()); }
	auto value_ = iter.unwrap()->next();
	if (value_.is_err()) { return Err(value_.unwrap_err()); }

	size_t index = 0;
	while (value_.is_ok()) {
		auto *value = value_.unwrap();
		if (value->type()->issubclass(types::tuple())) {
			const auto &other_pair = static_cast<const PyTuple &>(*value);
			if (other_pair.size() != 2) {
				return Err(value_error(
					"dictionary update sequence element #{} has length {}; 2 is required",
					index,
					other_pair.size()));
			}
			if (override) {
				m_map.insert_or_assign(other_pair.elements()[0], other_pair.elements()[1]);
			} else {
				m_map.insert({ other_pair.elements()[0], other_pair.elements()[1] });
			}
		} else {
			auto iter_inner = value->iter();
			if (iter_inner.is_err()) { return Err(iter_inner.unwrap_err()); }
			auto value_inner_ = iter_inner.unwrap()->next();
			if (value_inner_.is_err()) { return Err(value_inner_.unwrap_err()); }
			Value key = value_inner_.unwrap();
			value_inner_ = iter_inner.unwrap()->next();
			if (value_inner_.is_err()) { return Err(value_inner_.unwrap_err()); }
			Value value = value_inner_.unwrap();

			value_inner_ = iter_inner.unwrap()->next();
			if (value_inner_.is_ok()) {
				return Err(value_error(
					"dictionary update sequence element #{} does not have required length 2",
					index));
			}
			if (!value_inner_.unwrap_err()->type()->issubclass(types::stop_iteration())) {
				return Err(value_inner_.unwrap_err());
			}

			if (override) {
				m_map.insert_or_assign(std::move(key), std::move(value));
			} else {
				m_map.insert({ std::move(key), std::move(value) });
			}
			value_inner_ = iter_inner.unwrap()->next();
		}
		value_ = iter.unwrap()->next();
		index++;
	}

	if (!value_.unwrap_err()->type()->issubclass(types::stop_iteration())) {
		return Err(value_.unwrap_err());
	}

	return Ok(std::monostate{});
}

PyResult<PyObject *> PyDict::update(PyObject *other)
{
	if (other->type() == types::dict()) {
		if (auto result = merge(static_cast<PyDict *>(other), true); result.is_ok()) {
			return Ok(py_none());
		} else {
			return Err(result.unwrap_err());
		}
	}

	if (auto keys = other->getattribute(PyString::create("keys").unwrap()); keys.is_ok()) {
		if (auto result = merge(other, true); result.is_ok()) {
			return Ok(py_none());
		} else {
			return Err(result.unwrap_err());
		}
	}
	return merge_from_seq_2(other, true).and_then([](auto) -> PyResult<PyObject *> {
		return Ok(py_none());
	});
}

PyResult<std::monostate> PyDict::merge(PyDict *other, bool override)
{
	if (override) {
		for (const auto &[key, value] : other->map()) { m_map.insert_or_assign(key, value); }
	} else {
		for (const auto &[key, value] : other->map()) { m_map.insert({ key, value }); }
	}

	return Ok(std::monostate{});
}

PyResult<PyObject *> PyDict::setdefault(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyObject *, PyObject *>::unpack_tuple(args,
		kwargs,
		"dict.setdefault",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 2>{},
		py_none() /* default */);
	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [key, default_] = result.unwrap();

	if (const auto it = m_map.find(key); it != m_map.end()) { return PyObject::from(it->second); }

	insert(key, default_);

	return Ok(default_);
}


PyResult<PyObject *> PyDict::fromkeys(PyObject *iterable, PyObject *value)
{
	auto iterator = iterable->iter();
	if (iterator.is_err()) return iterator;

	auto map_ = PyDict::create();
	if (map_.is_err()) return map_;
	auto *map = map_.unwrap();

	if (!value) { value = py_none(); }

	auto key = iterator.unwrap()->next();
	while (key.is_ok()) {
		map->m_map.emplace(key.unwrap(), value);
		key = iterator.unwrap()->next();
	}

	if (key.unwrap_err() && !key.unwrap_err()->type()->issubclass(stop_iteration()->type())) {
		return key;
	}

	return Ok(map);
}

namespace {

	std::once_flag dict_flag;

	std::unique_ptr<TypePrototype> register_dict()
	{
		return std::move(
			klass<PyDict>("dict")
				.def(
					"get",
					+[](PyDict *self, PyTuple *args, PyDict *kwargs) {
						ASSERT(args)
						ASSERT(!kwargs || kwargs->size() == 0)
						auto key_ = PyObject::from(args->elements()[0]);
						if (key_.is_err()) return key_;
						PyObject *key = key_.unwrap();
						PyObject *default_value = nullptr;
						if (args->elements().size() == 2) {
							auto default_value_ = PyObject::from(args->elements()[1]);
							if (default_value_.is_err()) return default_value_;
							default_value = default_value_.unwrap();
						}
						return self->get(key, default_value);
					})
				.def(
					"pop",
					+[](PyDict *self, PyTuple *args, PyDict *kwargs) {
						ASSERT(args)
						ASSERT(!kwargs || kwargs->size() == 0)
						auto key_ = PyObject::from(args->elements()[0]);
						if (key_.is_err()) return key_;
						PyObject *key = key_.unwrap();
						PyObject *default_value = nullptr;
						if (args->elements().size() == 2) {
							auto default_value_ = PyObject::from(args->elements()[1]);
							if (default_value_.is_err()) return default_value_;
							default_value = default_value_.unwrap();
						}
						return self->pop(key, default_value);
					})
				.def(
					"update",
					+[](PyDict *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
						ASSERT(args)
						ASSERT(!kwargs || kwargs->size() == 0)
						auto other_ = PyObject::from(args->elements()[0]);
						if (other_.is_err()) return other_;
						auto *other = other_.unwrap();
						return self->update(other);
					})
				.def(
					"items",
					+[](PyDict *self, PyTuple *, PyDict *) -> PyResult<PyObject *> {
						return PyDictItems::create(*self);
					})
				.def("keys", &PyDict::keys)
				.def("values", &PyDict::values)
				.def("setdefault", &PyDict::setdefault)
				.classmethod(
					"fromkeys",
					+[](PyType *cls, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
						ASSERT(cls == types::dict());
						ASSERT(args && args->elements().size() > 0);
						ASSERT(!kwargs || kwargs->map().size());

						auto iterable_ = PyObject::from(args->elements()[0]);
						if (iterable_.is_err()) return iterable_;
						auto *iterable = iterable_.unwrap();

						auto value_ = [args]() -> PyResult<PyObject *> {
							if (args->elements().size() == 2) {
								return PyObject::from(args->elements()[1]);
							} else if (args->elements().size() == 3) {
								TODO();
							} else {
								return Ok(nullptr);
							}
						}();

						if (value_.is_err()) return value_;
						auto *value = value_.unwrap();

						return PyDict::fromkeys(iterable, value);
					})
				.type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyDict::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(dict_flag, []() { type = register_dict(); });
		return std::move(type);
	};
}

PyResult<PyDictItems *> PyDictItems::create(const PyDict &pydict)
{
	auto *result = VirtualMachine::the().heap().allocate<PyDictItems>(pydict);
	if (!result) { return Err(memory_error(sizeof(PyDictItems))); }
	return Ok(result);
}

PyDictItems::PyDictItems(PyType *type) : PyBaseObject(type) {}

PyDictItems::PyDictItems(const PyDict &pydict)
	: PyBaseObject(types::BuiltinTypes::the().dict_items()), m_pydict(pydict)
{}

PyResult<PyObject *> PyDictItems::__iter__() const { return PyDictItemsIterator::create(*this); }


PyDictItemsIterator PyDictItems::begin() const { return PyDictItemsIterator(*this); }

PyDictItemsIterator PyDictItems::end() const
{
	ASSERT(m_pydict);

	auto end_position = std::distance(m_pydict->get().map().begin(), m_pydict->get().map().end());
	return PyDictItemsIterator(*this, end_position);
}

std::string PyDictItems::to_string() const
{
	std::ostringstream os;
	os << "dict_items([";
	auto it = begin();
	auto end_it = end();

	while (it != end_it) {
		os << (*it)->to_string();
		std::advance(it, 1);
		if (it != end_it) { os << ","; }
	}

	os << "])";
	return os.str();
}

void PyDictItems::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(const_cast<PyDict &>(m_pydict->get()));
}

PyType *PyDictItems::static_type() const { return types::dict_items(); }

namespace {

	std::once_flag dict_items_flag;

	std::unique_ptr<TypePrototype> register_dict_items()
	{
		return std::move(klass<PyDictItems>("dict_items").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyDictItems::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(dict_items_flag, []() { type = register_dict_items(); });
		return std::move(type);
	};
}

// ---

PyResult<PyDictKeys *> PyDictKeys::create(const PyDict &pydict)
{
	auto *result = VirtualMachine::the().heap().allocate<PyDictKeys>(pydict);
	if (!result) { return Err(memory_error(sizeof(PyDictKeys))); }
	return Ok(result);
}

PyDictKeys::PyDictKeys(PyType *type) : PyBaseObject(type) {}

PyDictKeys::PyDictKeys(const PyDict &pydict)
	: PyBaseObject(types::BuiltinTypes::the().dict_keys()), m_pydict(pydict)
{}

PyResult<PyObject *> PyDictKeys::__iter__() const { return PyDictKeyIterator::create(*this); }

PyDictKeyIterator PyDictKeys::begin() const { return PyDictKeyIterator(*this); }

PyDictKeyIterator PyDictKeys::end() const
{
	ASSERT(m_pydict);
	auto end_position = std::distance(m_pydict->get().map().begin(), m_pydict->get().map().end());
	return PyDictKeyIterator(*this, end_position);
}

std::string PyDictKeys::to_string() const
{
	std::ostringstream os;
	os << "dict_keys([";
	auto it = begin();
	auto end_it = end();

	while (it != end_it) {
		os << (*it)->to_string();
		std::advance(it, 1);
		if (it != end_it) { os << ","; }
	}

	os << "])";
	return os.str();
}

void PyDictKeys::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(const_cast<PyDict &>(m_pydict->get()));
}

PyType *PyDictKeys::static_type() const { return types::dict_keys(); }

namespace {

	std::once_flag dict_keys_flag;

	std::unique_ptr<TypePrototype> register_dict_keys()
	{
		return std::move(klass<PyDictKeys>("dict_keys").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyDictKeys::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(dict_keys_flag, []() { type = register_dict_keys(); });
		return std::move(type);
	};
}

PyResult<PyDictValues *> PyDictValues::create(const PyDict &pydict)
{
	auto *result = VirtualMachine::the().heap().allocate<PyDictValues>(pydict);
	if (!result) { return Err(memory_error(sizeof(PyDictKeys))); }
	return Ok(result);
}

PyDictValues::PyDictValues(PyType *type) : PyBaseObject(type) {}

PyDictValues::PyDictValues(const PyDict &pydict)
	: PyBaseObject(types::BuiltinTypes::the().dict_values()), m_pydict(pydict)
{}

PyResult<PyObject *> PyDictValues::__repr__() const
{
	std::ostringstream os;

	[[maybe_unused]] struct Cleanup
	{
		const PyDictValues *dict_values;
		bool do_cleanup;

		~Cleanup()
		{
			if (do_cleanup) {
				auto it = visited_dict_values.find(const_cast<PyDictValues *>(dict_values));
				if (it != visited_dict_values.end()) { visited_dict_values.erase(it); }
			}
		}
	} cleanup{ this, !visited_dict_values.contains(const_cast<PyDictValues *>(this)) };
	visited_dict_values.insert(const_cast<PyDictValues *>(this));

	auto repr = [](const auto &el) -> PyResult<PyString *> {
		return std::visit(overloaded{
							  [](const auto &value) { return PyString::create(value.to_string()); },
							  [](PyObject *value) {
								  if (visited_dict_values.contains(value)) {
									  return PyString::create("...");
								  }
								  return value->repr();
							  },
						  },
			el);
	};
	os << "dict_values([";
	if (!m_pydict->get().map().empty()) {
		auto it = m_pydict->get().map().begin();
		while (std::next(it) != m_pydict->get().map().end()) {
			auto r = repr(it->second);
			if (r.is_err()) { return r; }
			os << std::move(r.unwrap()->value()) << ", ";
			std::advance(it, 1);
		}
		auto r = repr(it->second);
		if (r.is_err()) { return r; }
		os << std::move(r.unwrap()->value());
	}
	os << "])";

	return PyString::create(os.str());
}

PyResult<PyObject *> PyDictValues::__iter__() const { return PyDictValueIterator::create(*this); }

PyDictValueIterator PyDictValues::begin() const { return PyDictValueIterator(*this); }

PyDictValueIterator PyDictValues::end() const
{
	ASSERT(m_pydict);
	auto end_position = std::distance(m_pydict->get().map().begin(), m_pydict->get().map().end());
	return PyDictValueIterator(*this, end_position);
}

std::string PyDictValues::to_string() const
{
	std::ostringstream os;
	os << "dict_values([";
	auto it = begin();
	auto end_it = end();

	while (it != end_it) {
		os << (*it)->to_string();
		std::advance(it, 1);
		if (it != end_it) { os << ","; }
	}

	os << "])";
	return os.str();
}

void PyDictValues::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(const_cast<PyDict &>(m_pydict->get()));
}

PyType *PyDictValues::static_type() const { return types::dict_values(); }

namespace {

	std::once_flag dict_values_flag;

	std::unique_ptr<TypePrototype> register_dict_values()
	{
		return std::move(klass<PyDictValues>("dict_values").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyDictValues::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(dict_values_flag, []() { type = register_dict_values(); });
		return std::move(type);
	};
}


// iterators

PyDictItemsIterator::PyDictItemsIterator(PyType *type) : PyBaseObject(type) {}

PyDictItemsIterator::PyDictItemsIterator(const PyDictItems &pydict_items)
	: PyBaseObject(types::BuiltinTypes::the().dict_items_iterator()), m_pydictitems(pydict_items),
	  m_current_iterator(m_pydictitems->get().m_pydict->get().map().begin())
{}

PyDictItemsIterator::PyDictItemsIterator(const PyDictItems &pydict_items, size_t position)
	: PyDictItemsIterator(pydict_items)
{
	std::advance(m_current_iterator, position);
}


PyResult<PyDictItemsIterator *> PyDictItemsIterator::create(const PyDictItems &pydict_items)
{
	auto *result = VirtualMachine::the().heap().allocate<PyDictItemsIterator>(pydict_items);
	if (!result) { return Err(memory_error(sizeof(PyDictItemsIterator))); }
	return Ok(result);
}

PyResult<PyDictItemsIterator *> PyDictItemsIterator::create(const PyDictItems &pydict_items,
	size_t position)
{
	auto *result =
		VirtualMachine::the().heap().allocate<PyDictItemsIterator>(pydict_items, position);
	if (!result) { return Err(memory_error(sizeof(PyDictItemsIterator))); }
	return Ok(result);
}

void PyDictItemsIterator::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_current_iterator != m_pydictitems->get().m_pydict->get().map().end()) {
		if (std::holds_alternative<PyObject *>(m_current_iterator->first)) {
			visitor.visit(*std::get<PyObject *>(m_current_iterator->first));
		}
		if (std::holds_alternative<PyObject *>(m_current_iterator->second)) {
			visitor.visit(*std::get<PyObject *>(m_current_iterator->second));
		}
	}
	visitor.visit(const_cast<PyDictItems &>(m_pydictitems->get()));
}

std::string PyDictItemsIterator::to_string() const
{
	return fmt::format("<dict_itemiterator at {}>", static_cast<const void *>(this));
}

PyResult<PyObject *> PyDictItemsIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyDictItemsIterator::__next__()
{
	if (m_current_iterator != m_pydictitems->get().m_pydict->get().map().end()) {
		const auto &[key, value] = *m_current_iterator;
		m_current_iterator++;
		return PyTuple::create(key, value);
	}
	return Err(stop_iteration());
}

bool PyDictItemsIterator::operator==(const PyDictItemsIterator &other) const
{
	return m_pydictitems.has_value() && other.m_pydictitems.has_value()
		   && &m_pydictitems->get() == &other.m_pydictitems->get()
		   && m_current_iterator == other.m_current_iterator;
}

PyDictItemsIterator &PyDictItemsIterator::operator++()
{
	m_current_iterator++;
	return *this;
}

PyTuple *PyDictItemsIterator::operator*() const
{
	const auto &[key, value] = *m_current_iterator;
	return VirtualMachine::the().heap().allocate<PyTuple>(std::vector{ key, value });
}

PyType *PyDictItemsIterator::static_type() const { return types::dict_items_iterator(); }

namespace {

	std::once_flag dict_items_iterator_flag;

	std::unique_ptr<TypePrototype> register_dict_items_iterator()
	{
		return std::move(klass<PyDictItemsIterator>("dict_itemiterator").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyDictItemsIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(dict_items_iterator_flag, []() { type = register_dict_items_iterator(); });
		return std::move(type);
	};
}

PyDictKeyIterator::PyDictKeyIterator(PyType *type) : PyBaseObject(type) {}

PyDictKeyIterator::PyDictKeyIterator(const PyDictKeys &pydict_keys)
	: PyBaseObject(types::BuiltinTypes::the().dict_key_iterator()), m_pydictkeys(pydict_keys),
	  m_current_iterator(m_pydictkeys->get().m_pydict->get().map().begin())
{}

PyDictKeyIterator::PyDictKeyIterator(const PyDictKeys &pydict_keys, size_t position)
	: PyDictKeyIterator(pydict_keys)
{
	std::advance(m_current_iterator, position);
}


PyResult<PyDictKeyIterator *> PyDictKeyIterator::create(const PyDictKeys &pydict_keys)
{
	auto *result = VirtualMachine::the().heap().allocate<PyDictKeyIterator>(pydict_keys);
	if (!result) { return Err(memory_error(sizeof(PyDictKeyIterator))); }
	return Ok(result);
}

PyResult<PyDictKeyIterator *> PyDictKeyIterator::create(const PyDictKeys &pydict_keys,
	size_t position)
{
	auto *result = VirtualMachine::the().heap().allocate<PyDictKeyIterator>(pydict_keys, position);
	if (!result) { return Err(memory_error(sizeof(PyDictKeyIterator))); }
	return Ok(result);
}

void PyDictKeyIterator::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_current_iterator != m_pydictkeys->get().m_pydict->get().map().end()) {
		if (std::holds_alternative<PyObject *>(m_current_iterator->first)) {
			visitor.visit(*std::get<PyObject *>(m_current_iterator->first));
		}
	}
	visitor.visit(const_cast<PyDictKeys &>(m_pydictkeys->get()));
}

std::string PyDictKeyIterator::to_string() const
{
	return fmt::format("<dict_keyiterator at {}>", static_cast<const void *>(this));
}

PyResult<PyObject *> PyDictKeyIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyDictKeyIterator::__next__()
{
	if (m_current_iterator != m_pydictkeys->get().m_pydict->get().map().end()) {
		const auto &key = m_current_iterator->first;
		m_current_iterator++;
		return PyObject::from(key);
	}
	return Err(stop_iteration());
}

bool PyDictKeyIterator::operator==(const PyDictKeyIterator &other) const
{
	return m_pydictkeys.has_value() && other.m_pydictkeys.has_value()
		   && &m_pydictkeys->get() == &other.m_pydictkeys->get()
		   && m_current_iterator == other.m_current_iterator;
}

PyDictKeyIterator &PyDictKeyIterator::operator++()
{
	m_current_iterator++;
	return *this;
}

PyObject *PyDictKeyIterator::operator*() const
{
	const auto &key = m_current_iterator->first;
	return PyObject::from(key).unwrap();
}

PyType *PyDictKeyIterator::static_type() const { return types::dict_key_iterator(); }

namespace {

	std::once_flag dict_key_iterator_flag;

	std::unique_ptr<TypePrototype> register_dict_key_iterator()
	{
		return std::move(klass<PyDictKeyIterator>("dict_keyiterator").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyDictKeyIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(dict_key_iterator_flag, []() { type = register_dict_key_iterator(); });
		return std::move(type);
	};
}

PyDictValueIterator::PyDictValueIterator(PyType *type) : PyBaseObject(type) {}

PyDictValueIterator::PyDictValueIterator(const PyDictValues &pydict_values)
	: PyBaseObject(types::BuiltinTypes::the().dict_value_iterator()), m_pydictvalues(pydict_values),
	  m_current_iterator(m_pydictvalues->get().m_pydict->get().map().begin())
{}

PyDictValueIterator::PyDictValueIterator(const PyDictValues &pydict_values, size_t position)
	: PyDictValueIterator(pydict_values)
{
	std::advance(m_current_iterator, position);
}

PyResult<PyDictValueIterator *> PyDictValueIterator::create(const PyDictValues &pydict_values)
{
	auto *result = VirtualMachine::the().heap().allocate<PyDictValueIterator>(pydict_values);
	if (!result) { return Err(memory_error(sizeof(PyDictValueIterator))); }
	return Ok(result);
}

PyResult<PyDictValueIterator *> PyDictValueIterator::create(const PyDictValues &pydict_values,
	size_t position)
{
	auto *result =
		VirtualMachine::the().heap().allocate<PyDictValueIterator>(pydict_values, position);
	if (!result) { return Err(memory_error(sizeof(PyDictValueIterator))); }
	return Ok(result);
}

void PyDictValueIterator::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (std::holds_alternative<PyObject *>(m_current_iterator->second)) {
		visitor.visit(*std::get<PyObject *>(m_current_iterator->second));
	}

	visitor.visit(const_cast<PyDictValues &>(m_pydictvalues->get()));
}

std::string PyDictValueIterator::to_string() const
{
	return fmt::format("<dict_valueiterator at {}>", static_cast<const void *>(this));
}

PyResult<PyObject *> PyDictValueIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyDictValueIterator::__next__()
{
	if (m_current_iterator != m_pydictvalues->get().m_pydict->get().map().end()) {
		const auto &value = m_current_iterator->second;
		m_current_iterator++;
		return PyObject::from(value);
	}
	return Err(stop_iteration());
}

bool PyDictValueIterator::operator==(const PyDictValueIterator &other) const
{
	return m_pydictvalues.has_value() && other.m_pydictvalues.has_value()
		   && &m_pydictvalues->get() == &other.m_pydictvalues->get()
		   && m_current_iterator == other.m_current_iterator;
}

PyDictValueIterator &PyDictValueIterator::operator++()
{
	m_current_iterator++;
	return *this;
}

PyObject *PyDictValueIterator::operator*() const
{
	const auto &value = m_current_iterator->second;
	return PyObject::from(value).unwrap();
}

PyType *PyDictValueIterator::static_type() const { return types::dict_value_iterator(); }

namespace {

	std::once_flag dict_value_iterator_flag;

	std::unique_ptr<TypePrototype> register_dict_value_iterator()
	{
		return std::move(klass<PyDictValueIterator>("dict_valueiterator").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyDictValueIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(dict_value_iterator_flag, []() { type = register_dict_value_iterator(); });
		return std::move(type);
	};
}

}// namespace py
