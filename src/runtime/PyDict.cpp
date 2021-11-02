#include "bytecode/VM.hpp"
#include "interpreter/Interpreter.hpp"
#include "PyDict.hpp"
#include "PyString.hpp"
#include "StopIterationException.hpp"

std::string PyDict::to_string() const
{
	if (m_map.empty()) { return "{}"; }

	std::ostringstream os;
	os << "{";

	auto it = m_map.begin();
	while (std::next(it) != m_map.end()) {
		std::visit(
			overloaded{ [&os](PyObject *key) {
						   os << key->repr_impl(*VirtualMachine::the().interpreter())->to_string();
					   },
				[&os](const auto &key) { os << key; } },
			it->first);
		os << ": ";
		std::visit(
			overloaded{
				[&os, this](PyObject *value) {
					if (value == this) {
						os << "{...}";
					} else {
						os << value->repr_impl(*VirtualMachine::the().interpreter())->to_string();
					}
				},
				[&os](const auto &value) { os << value; } },
			it->second);
		os << ", ";

		std::advance(it, 1);
	}
	std::visit(
		overloaded{ [&os](PyObject *key) {
					   os << key->repr_impl(*VirtualMachine::the().interpreter())->to_string()
						  << ": ";
				   },
			[&os](const auto &key) { os << key << ": "; } },
		it->first);
	std::visit(
		overloaded{
			[&os, this](PyObject *value) {
				if (value == this) {
					os << "{...}";
				} else {
					os << value->repr_impl(*VirtualMachine::the().interpreter())->to_string();
				}
			},
			[&os](const auto &value) { os << value; } },
		it->second);
	os << "}";

	return os.str();
}

PyObject *PyDict::repr_impl(Interpreter &) const { return PyString::from(String{ to_string() }); }

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

PyObject *PyDictItemsIterator::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}

PyObject *PyDictItemsIterator::next_impl(Interpreter &interpreter)
{
	if (m_current_iterator != m_pydictitems.m_pydict.map().end()) {
		auto [key, value] = *m_current_iterator;
		m_current_iterator++;
		return VirtualMachine::the().heap().allocate<PyTuple>(std::vector{ key, value });
	}
	interpreter.raise_exception(stop_iteration(""));
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
