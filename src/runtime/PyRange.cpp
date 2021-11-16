#include "PyRange.hpp"
#include "PyNumber.hpp"
#include "PyString.hpp"
#include "StopIterationException.hpp"

#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

std::string PyRange::to_string() const
{
	if (m_step == 1) {
		return fmt::format("range({}, {})", m_start, m_stop);
	} else {
		return fmt::format("range({}, {}, {})", m_start, m_stop, m_step);
	}
}

PyObject *PyRange::repr_impl() const { return PyString::create(to_string()); }

PyObject *PyRange::iter_impl(Interpreter &) const
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyRangeIterator>(*this);
}

std::string PyRangeIterator::to_string() const
{
	return fmt::format("<range_iterator at {}>", static_cast<const void *>(this));
}

PyObject *PyRangeIterator::repr_impl() const { return PyString::create(to_string()); }

PyObject *PyRangeIterator::next_impl(Interpreter &interpreter)
{
	if (m_current_index < m_pyrange.stop()) {
		auto result = PyNumber::from(Number{ m_current_index });
		m_current_index += m_pyrange.step();
		return result;
	}
	interpreter.raise_exception(stop_iteration(""));
	return nullptr;
}
