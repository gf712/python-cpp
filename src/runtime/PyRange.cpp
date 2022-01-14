#include "PyRange.hpp"
#include "PyDict.hpp"
#include "PyInteger.hpp"
#include "PyString.hpp"
#include "StopIteration.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

PyObject *PyRange::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().size() == 0)
	ASSERT(args && args->size() > 0 && args->size() < 4)
	ASSERT(type == range())

	if (args->size() == 1) {
		auto stop = as<PyInteger>(PyObject::from(args->elements()[0]));
		return VirtualMachine::the().heap().allocate<PyRange>(
			std::get<int64_t>(stop->value().value));
	} else if (args->size() == 2) {
		auto start = as<PyInteger>(PyObject::from(args->elements()[0]));
		auto stop = as<PyInteger>(PyObject::from(args->elements()[1]));
		return VirtualMachine::the().heap().allocate<PyRange>(
			std::get<int64_t>(start->value().value), std::get<int64_t>(stop->value().value));
	} else if (args->size() == 3) {
		auto start = as<PyInteger>(PyObject::from(args->elements()[0]));
		auto stop = as<PyInteger>(PyObject::from(args->elements()[1]));
		auto step = as<PyInteger>(PyObject::from(args->elements()[2]));
		return VirtualMachine::the().heap().allocate<PyRange>(
			std::get<int64_t>(start->value().value),
			std::get<int64_t>(stop->value().value),
			std::get<int64_t>(step->value().value));
	}
	ASSERT_NOT_REACHED()
}


PyRange::PyRange(int64_t stop) : PyRange(0, stop, 1) {}

PyRange::PyRange(int64_t start, int64_t stop) : PyRange(start, stop, 1) {}

PyRange::PyRange(int64_t start, int64_t stop, int64_t step)
	: PyBaseObject(BuiltinTypes::the().range()), m_start(start), m_stop(stop), m_step(step)
{}

std::string PyRange::to_string() const
{
	if (m_step == 1) {
		return fmt::format("range({}, {})", m_start, m_stop);
	} else {
		return fmt::format("range({}, {}, {})", m_start, m_stop, m_step);
	}
}

PyObject *PyRange::__repr__() const { return PyString::create(to_string()); }

PyObject *PyRange::__iter__() const
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyRangeIterator>(*this);
}

PyType *PyRange::type() const { return range(); }

namespace {

std::once_flag range_flag;

std::unique_ptr<TypePrototype> register_range() { return std::move(klass<PyRange>("range").type); }
}// namespace

std::unique_ptr<TypePrototype> PyRange::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(range_flag, []() { type = ::register_range(); });
	return std::move(type);
}


PyRangeIterator::PyRangeIterator(const PyRange &pyrange)
	: PyBaseObject(BuiltinTypes::the().range_iterator()), m_pyrange(pyrange),
	  m_current_index(m_pyrange.start())
{}

std::string PyRangeIterator::to_string() const
{
	return fmt::format("<range_iterator at {}>", static_cast<const void *>(this));
}

PyObject *PyRangeIterator::__repr__() const { return PyString::create(to_string()); }

PyObject *PyRangeIterator::__next__()
{
	if (m_current_index < m_pyrange.stop()) {
		auto result = PyNumber::from(Number{ m_current_index });
		m_current_index += m_pyrange.step();
		return result;
	}
	VirtualMachine::the().interpreter().raise_exception(stop_iteration(""));
	return nullptr;
}

PyType *PyRangeIterator::type() const { return range_iterator(); }

namespace {

std::once_flag range_iterator_flag;

std::unique_ptr<TypePrototype> register_range_iterator()
{
	return std::move(klass<PyRangeIterator>("range_iterator").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyRangeIterator::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(range_iterator_flag, []() { type = ::register_range_iterator(); });
	return std::move(type);
}