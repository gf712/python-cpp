#include "PyRange.hpp"
#include "MemoryError.hpp"
#include "PyDict.hpp"
#include "PyInteger.hpp"
#include "PyString.hpp"
#include "StopIteration.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult PyRange::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().size() == 0)
	ASSERT(args && args->size() > 0 && args->size() < 4)
	ASSERT(type == range())

	auto obj = [&]() -> std::variant<PyRange *, PyResult> {
		if (args->size() == 1) {
			if (auto arg1 = PyObject::from(args->elements()[0]); arg1.is_ok()) {
				auto stop = as<PyInteger>(arg1.unwrap_as<PyObject>());
				return VirtualMachine::the().heap().allocate<PyRange>(
					std::get<int64_t>(stop->value().value));
			} else {
				return arg1;
			}
		} else if (args->size() == 2) {
			auto start_ = PyObject::from(args->elements()[0]);
			if (start_.is_err()) return start_;
			auto *start = as<PyInteger>(start_.unwrap_as<PyObject>());
			auto stop_ = PyObject::from(args->elements()[1]);
			if (stop_.is_err()) return stop_;
			auto *stop = as<PyInteger>(stop_.unwrap_as<PyObject>());
			return VirtualMachine::the().heap().allocate<PyRange>(
				std::get<int64_t>(start->value().value), std::get<int64_t>(stop->value().value));
		} else if (args->size() == 3) {
			auto start_ = PyObject::from(args->elements()[0]);
			if (start_.is_err()) return start_;
			auto *start = as<PyInteger>(start_.unwrap_as<PyObject>());
			auto stop_ = PyObject::from(args->elements()[1]);
			if (stop_.is_err()) return stop_;
			auto *stop = as<PyInteger>(stop_.unwrap_as<PyObject>());
			auto step_ = PyObject::from(args->elements()[2]);
			if (step_.is_err()) return step_;
			auto *step = as<PyInteger>(step_.unwrap_as<PyObject>());
			return VirtualMachine::the().heap().allocate<PyRange>(
				std::get<int64_t>(start->value().value),
				std::get<int64_t>(stop->value().value),
				std::get<int64_t>(step->value().value));
		}
		ASSERT_NOT_REACHED()
	}();

	if (std::holds_alternative<PyResult>(obj)) return std::get<PyResult>(obj);
	if (!std::get<PyRange *>(obj)) { return PyResult::Err(memory_error(sizeof(PyRange))); }
	return PyResult::Ok(std::get<PyRange *>(obj));
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

PyResult PyRange::__repr__() const { return PyString::create(to_string()); }

PyResult PyRange::__iter__() const
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate<PyRangeIterator>(*this);
	if (!obj) { return PyResult::Err(memory_error(sizeof(PyRangeIterator))); }
	return PyResult::Ok(obj);
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

PyResult PyRangeIterator::__repr__() const { return PyString::create(to_string()); }

PyResult PyRangeIterator::__next__()
{
	if (m_current_index < m_pyrange.stop()) {
		auto result = PyNumber::from(Number{ m_current_index });
		m_current_index += m_pyrange.step();
		return result;
	}
	return PyResult::Err(stop_iteration(""));
}

PyType *PyRangeIterator::type() const { return range_iterator(); }

void PyRangeIterator::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(const_cast<PyRange &>(m_pyrange));
}

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