#include "bytecode/VM.hpp"
#include "interpreter/Interpreter.hpp"
#include "PyDict.hpp"
#include "StopIterationException.hpp"

std::string PyDict::to_string() const
{
	if (m_map.empty()) { return "{}"; }

	std::ostringstream os;
	os << "{";

	auto it = m_map.begin();
	while (std::next(it) != m_map.end()) {
		std::visit([&os](const auto &key) { os << key << ": "; }, it->first);
		std::visit([&os](const auto &value) { os << value << ", "; }, it->second);

		std::advance(it, 1);
	}
	std::visit([&os](const auto &key) { os << key << ": "; }, it->first);
	std::visit([&os](const auto &value) { os << value; }, it->second);
	os << "}";

	return os.str();
}

std::shared_ptr<PyObject> PyDict::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}

std::shared_ptr<PyDictItems> PyDict::items() const
{
	return VirtualMachine::the().heap().allocate<PyDictItems>(shared_from_this_as<PyDict>());
}

PyDictItemsIterator PyDictItems::begin() const
{
	return PyDictItemsIterator(shared_from_this_as<PyDictItems>());
}


PyDictItemsIterator PyDictItems::end() const
{
	auto result = PyDictItemsIterator(shared_from_this_as<PyDictItems>());
	result.m_current_iterator = m_pydict->map().end();
	return result;
}

std::string PyDictItems::to_string() const
{
	std::ostringstream os;
	os << "dict_items([";
	auto it = begin();

	while (std::next(it) != end()) {
		os << (*it)->to_string() << ", ";
		std::advance(it, 1);
	}
	os << (*it)->to_string() << "])";
	return os.str();
}


std::string PyDictItemsIterator::to_string() const
{
	return fmt::format("<dict_itemiterator at {}>", static_cast<const void *>(this));
}

std::shared_ptr<PyObject> PyDictItemsIterator::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}

std::shared_ptr<PyObject> PyDictItemsIterator::next_impl(Interpreter &interpreter)
{
	if (m_current_iterator != m_pydictitems->m_pydict->map().end()) {
		auto [key, value] = *m_current_iterator;
		m_current_iterator++;
		return VirtualMachine::the().heap().allocate<PyTuple>(std::vector{ key, value });
	}
	interpreter.raise_exception(stop_iteration(""));
	return nullptr;
}

bool PyDictItemsIterator::operator==(const PyDictItemsIterator &other) const
{
	return m_pydictitems.get() == other.m_pydictitems.get()
		   && m_current_iterator == other.m_current_iterator;
}

PyDictItemsIterator &PyDictItemsIterator::operator++()
{
	m_current_iterator++;
	return *this;
}

std::shared_ptr<PyTuple> PyDictItemsIterator::operator*() const
{
	auto [key, value] = *m_current_iterator;
	return VirtualMachine::the().heap().allocate<PyTuple>(std::vector{ key, value });
}
