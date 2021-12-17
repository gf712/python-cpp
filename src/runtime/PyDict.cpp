#include "PyDict.hpp"
#include "PyBool.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "StopIterationException.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

PyDict::PyDict(MapType &&map) : PyBaseObject(BuiltinTypes::the().dict()), m_map(std::move(map)) {}
PyDict::PyDict(const MapType &map) : PyBaseObject(BuiltinTypes::the().dict()), m_map(map) {}
PyDict::PyDict() : PyBaseObject(BuiltinTypes::the().dict()) {}

PyDict *PyDict::create() { return VirtualMachine::the().heap().allocate<PyDict>(); }

std::string PyDict::to_string() const
{
	if (m_map.empty()) { return "{}"; }

	std::ostringstream os;
	os << "{";

	auto it = m_map.begin();
	while (std::next(it) != m_map.end()) {
		std::visit(overloaded{ [&os](PyObject *key) { os << key->repr()->to_string(); },
					   [&os](const auto &key) { os << key; } },
			it->first);
		os << ": ";
		std::visit(overloaded{ [&os, this](PyObject *value) {
								  if (value == this) {
									  os << "{...}";
								  } else {
									  os << value->repr()->to_string();
								  }
							  },
					   [&os](const auto &value) { os << value; } },
			it->second);
		os << ", ";

		std::advance(it, 1);
	}
	std::visit(overloaded{ [&os](PyObject *key) { os << key->repr()->to_string() << ": "; },
				   [&os](const auto &key) { os << key << ": "; } },
		it->first);
	std::visit(overloaded{ [&os, this](PyObject *value) {
							  if (value == this) {
								  os << "{...}";
							  } else {
								  os << value->repr()->to_string();
							  }
						  },
				   [&os](const auto &value) { os << value; } },
		it->second);
	os << "}";

	return os.str();
}

PyObject *PyDict::__repr__() const { return PyString::from(String{ to_string() }); }

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

namespace {

std::once_flag dict_flag;

std::unique_ptr<TypePrototype> register_dict() { return std::move(klass<PyDict>("dict").type); }
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
	if (it == end_it) {
		os << "])";
		return os.str();
	}
	std::advance(end_it, -1);

	while (true) {
		if (it == end_it) break;
		os << (*it)->to_string() << ", ";
		std::advance(it, 1);
	}
	os << (*it)->to_string() << "])";
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

PyObject *PyDictItemsIterator::__repr__() const { return PyString::create(to_string()); }

PyObject *PyDictItemsIterator::__next__()
{
	if (m_current_iterator != m_pydictitems.m_pydict.map().end()) {
		auto [key, value] = *m_current_iterator;
		m_current_iterator++;
		return VirtualMachine::the().heap().allocate<PyTuple>(std::vector{ key, value });
	}
	VirtualMachine::the().interpreter().raise_exception(stop_iteration(""));
	return nullptr;
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

template<> PyDict *as(PyObject *obj)
{
	if (obj->type() == dict()) { return static_cast<PyDict *>(obj); }
	return nullptr;
}

template<> const PyDict *as(const PyObject *obj)
{
	if (obj->type() == dict()) { return static_cast<const PyDict *>(obj); }
	return nullptr;
}