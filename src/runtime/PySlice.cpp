#include "PySlice.hpp"
#include "MemoryError.hpp"
#include "PyBool.hpp"
#include "PyInteger.hpp"
#include "PyNone.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "ValueError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PySlice *as(PyObject *obj)
{
	if (obj->type() == types::slice()) { return static_cast<PySlice *>(obj); }
	return nullptr;
}


template<> const PySlice *as(const PyObject *obj)
{
	if (obj->type() == types::slice()) { return static_cast<const PySlice *>(obj); }
	return nullptr;
}

PySlice::PySlice(PyType *type)
	: PyBaseObject(type), m_start(py_none()), m_stop(py_none()), m_step(py_none())
{}

PySlice::PySlice() : PySlice(py_none(), py_none(), py_none()) {}

PySlice::PySlice(PyObject *stop) : PySlice(py_none(), stop, py_none()) {}

PySlice::PySlice(PyObject *start, PyObject *stop, PyObject *step)
	: PyBaseObject(types::BuiltinTypes::the().slice()), m_start(start), m_stop(stop), m_step(step)
{}

PyResult<PySlice *> PySlice::create(int64_t stop)
{
	auto stop_obj = PyInteger::create(stop);
	if (stop_obj.is_err()) return Err(stop_obj.unwrap_err());

	return create(stop_obj.unwrap());
}

PyResult<PySlice *> PySlice::create(int64_t start, int64_t stop, int64_t end)
{
	auto start_obj = PyInteger::create(start);
	if (start_obj.is_err()) return Err(start_obj.unwrap_err());

	auto stop_obj = PyInteger::create(stop);
	if (stop_obj.is_err()) return Err(stop_obj.unwrap_err());

	auto end_obj = PyInteger::create(end);
	if (end_obj.is_err()) return Err(end_obj.unwrap_err());

	return create(start_obj.unwrap(), stop_obj.unwrap(), end_obj.unwrap());
}

PyResult<PySlice *> PySlice::create(PyObject *stop)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PySlice>(stop)) { return Ok(obj); }
	return Err(memory_error(sizeof(PySlice)));
}

PyResult<PySlice *> PySlice::create(PyObject *start, PyObject *stop, PyObject *end)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PySlice>(start, stop, end)) { return Ok(obj); }
	return Err(memory_error(sizeof(PySlice)));
}

PyResult<PyObject *> PySlice::__new__(const PyType *type, PyTuple *, PyDict *)
{
	ASSERT(type == types::slice());

	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PySlice>()) { return Ok(obj); }
	return Err(memory_error(sizeof(PySlice)));
}

PyResult<int32_t> PySlice::__init__(PyTuple *args, PyDict *)
{
	if (args->size() == 0) { return Err(type_error("slice expected at least 1 argument, got 0")); }
	if (args->size() > 3) {
		return Err(type_error("slice expected at most 3 arguments, got {}", args->size()));
	}

	if (args->size() == 1) {
		auto stop = PyObject::from(args->elements()[0]);
		if (stop.is_err()) return Err(stop.unwrap_err());
		m_stop = stop.unwrap();
	} else {
		auto start = PyObject::from(args->elements()[0]);
		if (start.is_err()) return Err(start.unwrap_err());
		auto stop = PyObject::from(args->elements()[1]);
		if (stop.is_err()) return Err(stop.unwrap_err());
		auto step = [args]() -> PyResult<PyObject *> {
			if (args->size() > 2) { return PyObject::from(args->elements()[2]); }
			return Ok(py_none());
		}();

		if (step.is_err()) return Err(step.unwrap_err());

		m_start = start.unwrap();
		m_stop = stop.unwrap();
		m_step = step.unwrap();
	}

	return Ok(0);
}

std::string PySlice::to_string() const
{
	ASSERT(m_start);
	ASSERT(m_stop);
	ASSERT(m_step);
	return fmt::format(
		"slice({}, {}, {})", m_start->to_string(), m_stop->to_string(), m_step->to_string());
}

PyResult<PyObject *> PySlice::__repr__() const { return PyString::create(to_string()); }

PyResult<int64_t> PySlice::__hash__() const { return Err(type_error("unhashable type: 'slice'")); }

PyResult<PyObject *> PySlice::__eq__(const PyObject *obj) const
{
	if (!as<PySlice>(obj)) { return Ok(py_false()); }
	if (auto result = m_start->richcompare(obj, RichCompare::Py_EQ); result.is_err()) {
		return result;
	} else {
		if (!result.unwrap()) return Ok(py_false());
	}

	if (auto result = m_stop->richcompare(obj, RichCompare::Py_EQ); result.is_err()) {
		return result;
	} else {
		if (!result.unwrap()) return Ok(py_false());
	}

	if (auto result = m_step->richcompare(obj, RichCompare::Py_EQ); result.is_err()) {
		return result;
	} else {
		if (!result.unwrap()) return Ok(py_false());
	}

	return Ok(py_true());
}

PyResult<PyObject *> PySlice::__lt__(const PyObject *obj) const
{
	if (!as<PySlice>(obj)) {
		return Err(type_error(
			"'<' not supported between instances of 'slice' and '{}'", obj->type()->name()));
	}
	if (auto result = m_start->richcompare(obj, RichCompare::Py_LT); result.is_err()) {
		return result;
	} else {
		if (!result.unwrap()) return Ok(py_false());
	}

	if (auto result = m_stop->richcompare(obj, RichCompare::Py_LT); result.is_err()) {
		return result;
	} else {
		if (!result.unwrap()) return Ok(py_false());
	}

	if (auto result = m_step->richcompare(obj, RichCompare::Py_LT); result.is_err()) {
		return result;
	} else {
		if (!result.unwrap()) return Ok(py_false());
	}

	return Ok(py_true());
}

PyResult<std::tuple<int64_t, int64_t, int64_t>> PySlice::get_indices(int64_t length) const
{
	if (!as<PyInteger>(m_start) && m_start != py_none()) {
		return Err(
			type_error("slice indices must be integers or None or have an __index__ method"));
	}
	if (!as<PyInteger>(m_stop) && m_stop != py_none()) {
		return Err(
			type_error("slice indices must be integers or None or have an __index__ method"));
	}
	if (!as<PyInteger>(m_step) && m_step != py_none()) {
		return Err(
			type_error("slice indices must be integers or None or have an __index__ method"));
	}

	auto step = [this]() {
		if (m_step == py_none()) {
			return int64_t{ 1 };
		} else {
			return as<PyInteger>(m_step)->as_i64();
		}
	}();

	if (step == 0) { return Err(value_error("slice step cannot be zero")); }

	const auto lower_upper = [step, length]() -> std::array<int64_t, 2> {
		if (step < 0) {
			return { -1, length - 1 };
		} else {
			return { 0, length };
		}
	}();

	const auto lower = std::get<0>(lower_upper);
	const auto upper = std::get<1>(lower_upper);

	const auto start = [this, step, upper, lower, length] {
		if (m_start == py_none()) {
			return step > 0 ? lower : upper;
		} else {
			auto start = as<PyInteger>(m_start)->as_i64();
			if (start < 0) {
				start += length;
				return std::min(start, lower);
			} else {
				return std::min(start, upper);
			}
		}
	}();

	const auto stop = [this, step, upper, lower, length] {
		if (m_stop == py_none()) {
			return step > 0 ? upper : lower;
		} else {
			auto stop = as<PyInteger>(m_stop)->as_i64();
			if (stop < 0) {
				stop += length;
				return std::min(stop, lower);
			} else {
				return std::min(stop, upper);
			}
		}
	}();

	return Ok(std::make_tuple(start, stop, step));
}

PyResult<std::tuple<int64_t, int64_t, int64_t>> PySlice::unpack() const
{
	if (!as<PyInteger>(m_start) && m_start != py_none()) {
		return Err(
			type_error("slice indices must be integers or None or have an __index__ method"));
	}
	if (!as<PyInteger>(m_stop) && m_stop != py_none()) {
		return Err(
			type_error("slice indices must be integers or None or have an __index__ method"));
	}
	if (!as<PyInteger>(m_step) && m_step != py_none()) {
		return Err(
			type_error("slice indices must be integers or None or have an __index__ method"));
	}

	const auto step = m_step == py_none() ? 1 : as<PyInteger>(m_step)->as_i64();
	if (step == 0) { return Err(value_error("slice step cannot be zero")); }

	const auto start = m_start == py_none() ? (step < 0 ? std::numeric_limits<int64_t>::max() : 0)
											: as<PyInteger>(m_start)->as_i64();
	const auto stop = m_stop == py_none() ? (step < 0 ? std::numeric_limits<int64_t>::min()
													  : std::numeric_limits<int64_t>::max())
										  : as<PyInteger>(m_stop)->as_i64();

	return Ok(std::make_tuple(start, stop, step));
}

std::tuple<int64_t, int64_t, int64_t>
	PySlice::adjust_indices(int64_t start, int64_t stop, int64_t step, int64_t length)
{
	// directly copied from cpython, since, in their own words, "this is harder to get right than
	// you might think"
	ASSERT(step != 0);
	if (start < 0) {
		start += length;
		if (start < 0) { start = (step < 0) ? -1 : 0; }
	} else if (start >= length) {
		start = (step < 0) ? length - 1 : length;
	}

	if (stop < 0) {
		stop += length;
		if (stop < 0) { stop = (step < 0) ? -1 : 0; }
	} else if (stop >= length) {
		stop = (step < 0) ? length - 1 : length;
	}

	if (step < 0) {
		if (stop < start) { return { start, stop, (start - stop - 1) / (-step) + 1 }; }
	} else {
		if (start < stop) { return { start, stop, (stop - start - 1) / step + 1 }; }
	}
	return { start, stop, 0 };
}


namespace {

	std::once_flag slice_flag;

	std::unique_ptr<TypePrototype> register_slice()
	{
		return std::move(klass<PySlice>("slice")
							 .attr("start", &PySlice::m_start)
							 .attr("stop", &PySlice::m_stop)
							 .attr("step", &PySlice::m_step)
							 .type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PySlice::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(slice_flag, []() { type = register_slice(); });
		return std::move(type);
	};
}

PyType *PySlice::static_type() const { return types::slice(); }

void PySlice::visit_graph(Visitor &visitor)
{
	if (m_start) visitor.visit(*m_start);
	if (m_stop) visitor.visit(*m_stop);
	if (m_step) visitor.visit(*m_step);
}

}// namespace py
