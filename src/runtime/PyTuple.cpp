#include "PyTuple.hpp"
#include "PyString.hpp"
#include "StopIterationException.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

PyTuple *PyTuple::create()
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyTuple>();
}

PyTuple *PyTuple::create(std::vector<Value> elements)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyTuple>(std::move(elements));
}

PyTuple *PyTuple::create(const std::vector<PyObject *> &elements)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyTuple>(elements);
}

std::string PyTuple::to_string() const
{
	std::ostringstream os;

	os << "(";
	auto it = m_elements.begin();
	while (std::next(it) != m_elements.end()) {
		std::visit([&os](const auto &value) { os << value << ", "; }, *it);
		std::advance(it, 1);
	}
	std::visit([&os](const auto &value) { os << value; }, *it);
	os << ")";

	return os.str();
}

PyObject *PyTuple::repr_impl() const { return PyString::create(to_string()); }

PyObject *PyTuple::iter_impl(Interpreter &) const
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyTupleIterator>(*PyTuple::create());
}


PyTupleIterator PyTuple::begin() const { return PyTupleIterator(*this); }

PyTupleIterator PyTuple::end() const { return PyTupleIterator(*this, m_elements.size()); }

PyObject *PyTuple::operator[](size_t idx) const
{
	return std::visit([](const auto &value) { return PyObject::from(value); }, m_elements[idx]);
}

void PyTuple::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto &el : m_elements) {
		if (std::holds_alternative<PyObject *>(el)) {
			if (std::get<PyObject *>(el) != this) std::get<PyObject *>(el)->visit_graph(visitor);
		}
	}
}


std::string PyTupleIterator::to_string() const
{
	return fmt::format("<tuple_iterator at {}>", static_cast<const void *>(this));
}

PyObject *PyTupleIterator::repr_impl() const { return PyString::create(to_string()); }

PyObject *PyTupleIterator::next_impl(Interpreter &interpreter)
{
	if (m_current_index < m_pytuple.elements().size())
		return std::visit([](const auto &element) { return PyObject::from(element); },
			m_pytuple.elements()[m_current_index++]);
	interpreter.raise_exception(stop_iteration(""));
	return nullptr;
}

bool PyTupleIterator::operator==(const PyTupleIterator &other) const
{
	return &m_pytuple == &other.m_pytuple && m_current_index == other.m_current_index;
}

PyTupleIterator &PyTupleIterator::operator++()
{
	m_current_index++;
	return *this;
}

PyTupleIterator &PyTupleIterator::operator--()
{
	m_current_index--;
	return *this;
}

PyObject *PyTupleIterator::operator*() const
{
	return std::visit([](const auto &element) { return PyObject::from(element); },
		m_pytuple.elements()[m_current_index]);
}
