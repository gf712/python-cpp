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

PyResult<PyObject *> PyRange::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().size() == 0)
	ASSERT(args && args->size() > 0 && args->size() < 4)
	ASSERT(type == range())

	auto obj = [&]() -> std::variant<PyRange *, PyResult<PyRange *>> {
		if (args->size() == 1) {
			if (auto arg1 = PyObject::from(args->elements()[0]); arg1.is_ok()) {
				auto stop = as<PyInteger>(arg1.unwrap());
				return VirtualMachine::the().heap().allocate<PyRange>(stop);
			} else {
				return Err(arg1.unwrap_err());
			}
		} else if (args->size() == 2) {
			auto start_ = PyObject::from(args->elements()[0]);
			if (start_.is_err()) return Err(start_.unwrap_err());
			auto *start = as<PyInteger>(start_.unwrap());
			auto stop_ = PyObject::from(args->elements()[1]);
			if (stop_.is_err()) return Err(stop_.unwrap_err());
			auto *stop = as<PyInteger>(stop_.unwrap());
			return VirtualMachine::the().heap().allocate<PyRange>(start, stop);
		} else if (args->size() == 3) {
			auto start_ = PyObject::from(args->elements()[0]);
			if (start_.is_err()) return Err(start_.unwrap_err());
			auto *start = as<PyInteger>(start_.unwrap());
			auto stop_ = PyObject::from(args->elements()[1]);
			if (stop_.is_err()) return Err(stop_.unwrap_err());
			auto *stop = as<PyInteger>(stop_.unwrap());
			auto step_ = PyObject::from(args->elements()[2]);
			if (step_.is_err()) return Err(step_.unwrap_err());
			auto *step = as<PyInteger>(step_.unwrap());
			return VirtualMachine::the().heap().allocate<PyRange>(start, stop, step);
		}
		ASSERT_NOT_REACHED();
	}();

	if (std::holds_alternative<PyResult<PyRange *>>(obj)) return std::get<PyResult<PyRange *>>(obj);
	if (!std::get<PyRange *>(obj)) { return Err(memory_error(sizeof(PyRange))); }
	return Ok(std::get<PyRange *>(obj));
}


PyRange::PyRange(PyInteger *stop)
	: PyBaseObject(BuiltinTypes::the().range()), m_stop(std::get<int64_t>(stop->value().value))
{}

PyRange::PyRange(PyInteger *start, PyInteger *stop)
	: PyBaseObject(BuiltinTypes::the().range()), m_start(std::get<int64_t>(start->value().value)),
	  m_stop(std::get<int64_t>(stop->value().value))
{}

PyRange::PyRange(PyInteger *start, PyInteger *stop, PyInteger *step)
	: PyBaseObject(BuiltinTypes::the().range()), m_start(std::get<int64_t>(start->value().value)),
	  m_stop(std::get<int64_t>(stop->value().value)), m_step(std::get<int64_t>(step->value().value))
{}

std::string PyRange::to_string() const
{
	if (m_step == 1) {
		return fmt::format("range({}, {})", m_start, m_stop);
	} else {
		return fmt::format("range({}, {}, {})", m_start, m_stop, m_step);
	}
}

PyResult<PyObject *> PyRange::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyRange::__iter__() const
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate<PyRangeIterator>(*this);
	if (!obj) { return Err(memory_error(sizeof(PyRangeIterator))); }
	return Ok(obj);
}

PyType *PyRange::type() const { return range(); }

namespace {

std::once_flag range_flag;

std::unique_ptr<TypePrototype> register_range() { return std::move(klass<PyRange>("range").type); }
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyRange::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(range_flag, []() { type = ::register_range(); });
		return std::move(type);
	};
}


PyRangeIterator::PyRangeIterator(const PyRange &pyrange)
	: PyBaseObject(BuiltinTypes::the().range_iterator()), m_pyrange(pyrange),
	  m_current_index(m_pyrange.start())
{}

std::string PyRangeIterator::to_string() const
{
	return fmt::format("<range_iterator at {}>", static_cast<const void *>(this));
}

PyResult<PyObject *> PyRangeIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyRangeIterator::__next__()
{
	if (m_current_index < m_pyrange.stop()) {
		auto result = PyNumber::from(Number{ m_current_index });
		m_current_index += m_pyrange.step();
		return result;
	}
	return Err(stop_iteration());
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

std::function<std::unique_ptr<TypePrototype>()> PyRangeIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(range_iterator_flag, []() { type = ::register_range_iterator(); });
		return std::move(type);
	};
}