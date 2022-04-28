#include "PyDict.hpp"
#include "MemoryError.hpp"
#include "PyBool.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "RuntimeError.hpp"
#include "StopIteration.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

template<> PyDict *py::as(PyObject *obj)
{
	if (obj->type() == dict()) { return static_cast<PyDict *>(obj); }
	return nullptr;
}

template<> const PyDict *py::as(const PyObject *obj)
{
	if (obj->type() == dict()) { return static_cast<const PyDict *>(obj); }
	return nullptr;
}

PyDict::PyDict(MapType &&map) : PyBaseObject(BuiltinTypes::the().dict()), m_map(std::move(map)) {}

PyDict::PyDict(const MapType &map) : PyBaseObject(BuiltinTypes::the().dict()), m_map(map) {}

PyDict::PyDict() : PyBaseObject(BuiltinTypes::the().dict()) {}

PyResult PyDict::create()
{
	auto *result = VirtualMachine::the().heap().allocate<PyDict>();
	if (!result) { return PyResult::Err(memory_error(sizeof(PyDict))); }
	return PyResult::Ok(result);
}

PyResult PyDict::create(MapType &&map)
{
	auto *result = VirtualMachine::the().heap().allocate<PyDict>(std::move(map));
	if (!result) { return PyResult::Err(memory_error(sizeof(PyDict))); }
	return PyResult::Ok(result);
}

PyResult PyDict::create(const MapType &map)
{
	auto *result = VirtualMachine::the().heap().allocate<PyDict>(map);
	if (!result) { return PyResult::Err(memory_error(sizeof(PyDict))); }
	return PyResult::Ok(result);
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
								  os << r.unwrap_as<PyObject>()->to_string();
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
									  os << r.unwrap_as<PyObject>()->to_string();
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
							  os << r.unwrap_as<PyObject>()->to_string() << ": ";
						  },
				   [&os](const auto &key) { os << key << ": "; } },
		it->first);
	std::visit(overloaded{ [&os, this](PyObject *value) {
							  if (value == this) {
								  os << "{...}";
							  } else {
								  auto r = value->repr();
								  ASSERT(r.is_ok())
								  os << r.unwrap_as<PyObject>()->to_string();
							  }
						  },
				   [&os](const auto &value) { os << value; } },
		it->second);
	os << "}";

	return os.str();
}

PyResult PyDict::__repr__() const { return PyString::create(to_string()); }

PyResult PyDict::__eq__(const PyObject *other) const
{
	if (!as<PyDict>(other)) { return PyResult::Ok(py_false()); }

	return PyResult::Ok(m_map == as<PyDict>(other)->map() ? py_true() : py_false());
}

PyDictItems *PyDict::items() const
{
	return VirtualMachine::the().heap().allocate<PyDictItems>(*this);
}

Value PyDict::operator[](Value key) const
{
	if (auto iter = m_map.find(key); iter != m_map.end()) {
		return iter->second;
	} else {
		return py_none();
	}
}

void PyDict::insert(const Value &key, const Value &value) { m_map.insert_or_assign(key, value); }

PyResult PyDict::merge(PyTuple *args, PyDict *kwargs)
{
	ASSERT(args && args->size() == 1)
	ASSERT(!kwargs || kwargs->map().empty())

	auto other_dict_ = PyObject::from(args->elements()[0]);
	if (other_dict_.is_err()) return other_dict_;
	auto *other_dict = other_dict_.unwrap_as<PyObject>();
	ASSERT(as<PyDict>(other_dict))

	auto map_copy = as<PyDict>(other_dict)->map();
	m_map.merge(map_copy);
	if (!map_copy.empty()) {
		// should raise error if duplicates are not allowed
		TODO();
	}

	return PyResult::Ok(py_none());
}


void PyDict::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto &[key, value] : m_map) {
		if (std::holds_alternative<PyObject *>(value)) {
			if (std::get<PyObject *>(value) != this)
				std::get<PyObject *>(value)->visit_graph(visitor);
		}
		if (std::holds_alternative<PyObject *>(key)) {
			if (std::get<PyObject *>(key) != this) std::get<PyObject *>(key)->visit_graph(visitor);
		}
	}
}

PyType *PyDict::type() const { return ::dict(); }

PyResult PyDict::get(PyObject *key, PyObject *default_value) const
{
	if (auto it = m_map.find(key); it != m_map.end()) {
		return PyObject::from(it->second);
	} else if (default_value) {
		return PyResult::Ok(default_value);
	}
	return PyResult::Ok(py_none());
}

PyResult PyDict::update(PyDict *other)
{
	for (const auto &[key, value] : other->map()) {
		if (auto it = m_map.find(key); it != m_map.end()) {
			it->second = value;
		} else {
			other->insert(key, value);
		}
	}

	return PyResult::Ok(py_none());
}

namespace {

std::once_flag dict_flag;

std::unique_ptr<TypePrototype> register_dict()
{
	return std::move(klass<PyDict>("dict")
						 .def(
							 "get",
							 +[](PyDict *self, PyTuple *args, PyDict *kwargs) {
								 ASSERT(args)
								 ASSERT(!kwargs || kwargs->size() == 0)
								 auto key_ = PyObject::from(args->elements()[0]);
								 if (key_.is_err()) return key_;
								 PyObject *key = key_.unwrap_as<PyObject>();
								 PyObject *default_value = nullptr;
								 if (args->elements().size() == 2) {
									 auto default_value_ = PyObject::from(args->elements()[1]);
									 if (default_value_.is_err()) return default_value_;
									 default_value = default_value_.unwrap_as<PyObject>();
								 }
								 return self->get(key, default_value);
							 })
						 .def(
							 "update",
							 +[](PyDict *self, PyTuple *args, PyDict *kwargs) {
								 ASSERT(args)
								 ASSERT(!kwargs || kwargs->size() == 0)
								 auto other_ = PyObject::from(args->elements()[0]);
								 if (other_.is_err()) return other_;
								 auto *other = other_.unwrap_as<PyObject>();
								 if (other->type() != dict()) {
									 return PyResult::Err(runtime_error("TODO"));
								 }
								 return self->update(as<PyDict>(other));
							 })
						 .type);
}
}// namespace

std::unique_ptr<TypePrototype> PyDict::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(dict_flag, []() { type = ::register_dict(); });
	return std::move(type);
}

PyDictItems::PyDictItems(const PyDict &pydict)
	: PyBaseObject(BuiltinTypes::the().dict_items()), m_pydict(pydict)
{}

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
	visitor.visit(*const_cast<PyDict *>(&m_pydict));
}

PyType *PyDictItems::type() const { return dict_items(); }

namespace {

std::once_flag dict_items_flag;

std::unique_ptr<TypePrototype> register_dict_items()
{
	return std::move(klass<PyDictItems>("dict_items").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyDictItems::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(dict_items_flag, []() { type = ::register_dict_items(); });
	return std::move(type);
}

PyDictItemsIterator::PyDictItemsIterator(const PyDictItems &pydict)
	: PyBaseObject(BuiltinTypes::the().dict_items_iterator()), m_pydictitems(pydict),
	  m_current_iterator(m_pydictitems.m_pydict.map().begin())
{}

PyDictItemsIterator::PyDictItemsIterator(const PyDictItems &pydict, size_t position)
	: PyDictItemsIterator(pydict)
{
	std::advance(m_current_iterator, position);
}

PyDictItemsIterator PyDictItems::begin() const { return PyDictItemsIterator(*this); }

PyDictItemsIterator PyDictItems::end() const
{
	auto end_position = std::distance(m_pydict.map().begin(), m_pydict.map().end());
	return PyDictItemsIterator(*this, end_position);
}

void PyDictItemsIterator::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (std::holds_alternative<PyObject *>(m_current_iterator->first)) {
		visitor.visit(*std::get<PyObject *>(m_current_iterator->first));
	}
	if (std::holds_alternative<PyObject *>(m_current_iterator->second)) {
		visitor.visit(*std::get<PyObject *>(m_current_iterator->second));
	}
	visitor.visit(*const_cast<PyDictItems *>(&m_pydictitems));
}

std::string PyDictItemsIterator::to_string() const
{
	return fmt::format("<dict_itemiterator at {}>", static_cast<const void *>(this));
}

PyResult PyDictItemsIterator::__repr__() const { return PyString::create(to_string()); }

PyResult PyDictItemsIterator::__next__()
{
	if (m_current_iterator != m_pydictitems.m_pydict.map().end()) {
		auto [key, value] = *m_current_iterator;
		m_current_iterator++;
		return PyTuple::create(key, value);
	}
	return PyResult::Err(stop_iteration(""));
}

bool PyDictItemsIterator::operator==(const PyDictItemsIterator &other) const
{
	return &m_pydictitems == &other.m_pydictitems && m_current_iterator == other.m_current_iterator;
}

PyDictItemsIterator &PyDictItemsIterator::operator++()
{
	m_current_iterator++;
	return *this;
}

PyTuple *PyDictItemsIterator::operator*() const
{
	auto [key, value] = *m_current_iterator;
	return VirtualMachine::the().heap().allocate<PyTuple>(std::vector{ key, value });
}

PyType *PyDictItemsIterator::type() const { return dict_items_iterator(); }

namespace {

std::once_flag dict_items_iterator_flag;

std::unique_ptr<TypePrototype> register_dict_items_iterator()
{
	return std::move(klass<PyDictItemsIterator>("dict_itemiterator").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyDictItemsIterator::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(dict_items_iterator_flag, []() { type = ::register_dict_items_iterator(); });
	return std::move(type);
}
