#include "PyNumber.hpp"
#include "PyRange.hpp"
#include "PyString.hpp"
#include "StopIterationException.hpp"

#include "interpreter/Interpreter.hpp"
#include "bytecode/VM.hpp"

std::string PyRange::to_string() const
{
	if (m_step == 1) {
		return fmt::format("range({}, {})", m_start, m_stop);
	} else {
		return fmt::format("range({}, {}, {})", m_start, m_stop, m_step);
	}
}

std::shared_ptr<PyObject> PyRange::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}

std::shared_ptr<PyObject> PyRange::iter_impl(Interpreter &) const
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyRangeIterator>(shared_from_this_as<PyRange>());
}


std::string PyRangeIterator::to_string() const
{
	return fmt::format("<range_iterator at {}>", static_cast<const void *>(this));
}

std::shared_ptr<PyObject> PyRangeIterator::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}

std::shared_ptr<PyObject> PyRangeIterator::next_impl(Interpreter &interpreter)
{
	if (m_current_index < m_pyrange->stop()) {
		auto result = PyNumber::from(Number{ m_current_index });
		m_current_index += m_pyrange->step();
		return result;
	}
	interpreter.raise_exception(stop_iteration(""));
	return nullptr;
}
