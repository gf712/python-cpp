#include "PyRange.hpp"
#include "MemoryError.hpp"
#include "PyArgParser.hpp"
#include "PyDict.hpp"
#include "PyInteger.hpp"
#include "PyString.hpp"
#include "StopIteration.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/IndexError.hpp"
#include "runtime/PySlice.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/Value.hpp"
#include "runtime/ValueError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

PyRange::PyRange(PyType *type) : PyBaseObject(type) {}

PyResult<PyObject *> PyRange::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::range());

	auto result = PyArgsParser<PyObject *, PyObject *, PyObject *>::unpack_tuple(args,
		kwargs,
		"range",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 3>{},
		nullptr /* stop / start */,
		nullptr /* step */);
	if (result.is_err()) return Err(result.unwrap_err());
	auto [arg0, arg1, arg2] = result.unwrap();

	auto as_index = [](PyObject *obj) -> PyResult<PyInteger *> {
		if (auto *value = as<PyInteger>(obj)) { return Ok(value); }
		return Err(
			type_error("'{}' object cannot be interpreted as an integer", obj->type()->name()));
	};

	PyRange *range = nullptr;
	if (!arg1) {
		auto stop = as_index(arg0);
		if (stop.is_err()) return Err(stop.unwrap_err());
		range = VirtualMachine::the().heap().allocate<PyRange>(stop.unwrap());
	} else if (!arg2) {
		auto start = as_index(arg0);
		if (start.is_err()) return Err(start.unwrap_err());
		auto stop = as_index(arg1);
		if (stop.is_err()) return Err(stop.unwrap_err());
		range = VirtualMachine::the().heap().allocate<PyRange>(start.unwrap(), stop.unwrap());
	} else {
		auto start = as_index(arg0);
		if (start.is_err()) return Err(start.unwrap_err());
		auto stop = as_index(arg1);
		if (stop.is_err()) return Err(stop.unwrap_err());
		auto step = as_index(arg2);
		if (step.is_err()) return Err(step.unwrap_err());
		range = VirtualMachine::the().heap().allocate<PyRange>(
			start.unwrap(), stop.unwrap(), step.unwrap());
	}
	if (!range) { return Err(memory_error(sizeof(PyRange))); }
	return Ok(range);
}

PyRange::PyRange(BigIntType start, BigIntType stop, BigIntType step)
	: PyBaseObject(types::BuiltinTypes::the().range()), m_start(start), m_stop(stop), m_step(step)
{}

PyRange::PyRange(PyInteger *stop)
	: PyBaseObject(types::BuiltinTypes::the().range()),
	  m_stop(std::get<BigIntType>(stop->value().value))
{}

PyRange::PyRange(PyInteger *start, PyInteger *stop)
	: PyBaseObject(types::BuiltinTypes::the().range()),
	  m_start(std::get<BigIntType>(start->value().value)),
	  m_stop(std::get<BigIntType>(stop->value().value))
{}

PyRange::PyRange(PyInteger *start, PyInteger *stop, PyInteger *step)
	: PyBaseObject(types::BuiltinTypes::the().range()),
	  m_start(std::get<BigIntType>(start->value().value)),
	  m_stop(std::get<BigIntType>(stop->value().value)),
	  m_step(std::get<BigIntType>(step->value().value))
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

PyResult<PyObject *> PyRange::__reversed__() const
{
	// reversed(range(start, stop, step)) -> range(start+(n-1)*step, start-step, -step)
	// where n is the number of integers in the range.
	const BigIntType n = (m_stop - m_start) / m_step;
	BigIntType start = m_start + (n - 1) * m_step;
	BigIntType stop = m_start - m_step;
	BigIntType step = -m_step;

	auto *range = VirtualMachine::the().heap().allocate<PyRange>(start, stop, step);
	if (!range) { return Err(memory_error(sizeof(PyRange))); }

	return range->__iter__();
}

PyResult<PyObject *> PyRange::__getitem__(int64_t index_) const
{
	BigIntType index = index_;
	const BigIntType n = (m_stop - m_start) / m_step;
	if (index < 0) { index += n; }
	if (index >= n) { return Err(index_error("range object index out of range")); }
	return PyInteger::create(m_start + m_step * index);
}

PyResult<PyObject *> PyRange::__getitem__(PyObject *key) const
{
	if (key->type()->issubclass(types::integer())) {
		auto value = static_cast<const PyInteger &>(*key).as_big_int();
		if (!value.fits_slong_p()) { return Err(value_error("range object index too large")); }
		return __getitem__(value.get_si());
	} else if (key->type()->issubclass(types::slice())) {
		const auto &slice = static_cast<const PySlice &>(*key);
		auto slice_values = slice.unpack();
		if (slice_values.is_err()) { return Err(slice_values.unwrap_err()); }
		auto [start, stop, step] = slice_values.unwrap();

		start = start == std::numeric_limits<int64_t>::max() ? -1 : start;
		stop = (stop == std::numeric_limits<int64_t>::min()
				   || stop == std::numeric_limits<int64_t>::max())
				   ? -1
				   : stop;

		BigIntType new_step = step * m_step;
		BigIntType new_start = m_start + (start * m_step);
		BigIntType new_stop = m_start + (stop * m_step);

		auto *obj = VirtualMachine::the().heap().allocate<PyRange>(new_start, new_stop, new_step);
		if (!obj) { return Err(memory_error(sizeof(PyRange))); }
		return Ok(obj);
	}

	return Err(type_error("range indices must be integers or slices, not {}", key->type()->name()));
}


PyType *PyRange::static_type() const { return types::range(); }

namespace {

	std::once_flag range_flag;

	std::unique_ptr<TypePrototype> register_range()
	{
		return std::move(klass<PyRange>("range").def("__reversed__", &PyRange::__reversed__).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyRange::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(range_flag, []() { type = register_range(); });
		return std::move(type);
	};
}


PyRangeIterator::PyRangeIterator(const PyRange &pyrange)
	: PyBaseObject(types::BuiltinTypes::the().range_iterator()), m_pyrange(pyrange),
	  m_current_index(m_pyrange.start())
{}

std::string PyRangeIterator::to_string() const
{
	return fmt::format("<range_iterator at {}>", static_cast<const void *>(this));
}

PyResult<PyObject *> PyRangeIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyRangeIterator::__next__()
{
	auto within_range = [this](const BigIntType &current) {
		if (m_pyrange.step() < 0) {
			return current > m_pyrange.stop();
		} else {
			return current < m_pyrange.stop();
		}
	};
	if (within_range(m_current_index)) {
		auto result = PyNumber::from(Number{ m_current_index });
		m_current_index += m_pyrange.step();
		return result;
	}
	return Err(stop_iteration());
}

PyResult<PyObject *> PyRangeIterator::__iter__() const
{
	return Ok(const_cast<PyRangeIterator *>(this));
}

PyType *PyRangeIterator::static_type() const { return types::range_iterator(); }

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
		std::call_once(range_iterator_flag, []() { type = register_range_iterator(); });
		return std::move(type);
	};
}
}// namespace py
