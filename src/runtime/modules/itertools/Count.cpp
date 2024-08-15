#include "Count.hpp"
#include "runtime/MemoryError.hpp"
#include "runtime/PyArgParser.hpp"
#include "runtime/PyFloat.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/Value.hpp"
#include "runtime/forward.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"
#include "utilities.hpp"
#include <variant>

namespace py {
namespace {
	static PyType *s_itertools_count = nullptr;
}
namespace itertools {
	Count::Count(PyType *type) : PyBaseObject(type) {}

	Count::Count() : Count(s_itertools_count) {}

	Count::Count(Number start) : Count(s_itertools_count)
	{
		m_start = std::move(start);
		m_current = m_start;
	}

	Count::Count(Number start, Number step) : Count(s_itertools_count)
	{
		m_start = std::move(start);
		m_step = std::move(step);
		m_current = m_start;
	}

	PyResult<PyObject *> Count::create() { return create(Number{ 0 }, Number{ 1 }); }

	PyResult<PyObject *> Count::create(Number start)
	{
		return create(std::move(start), Number{ 1 });
	}

	PyResult<PyObject *> Count::create(Number start, Number step)
	{
		auto *obj = VirtualMachine::the().heap().allocate<Count>(std::move(start), std::move(step));
		if (!obj) { return Err(memory_error(sizeof(Count))); }
		return Ok(obj);
	}

	PyResult<PyObject *> Count::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
	{
		ASSERT(s_itertools_count);
		ASSERT(type == s_itertools_count);

		auto parsed_args = PyArgsParser<PyObject *, PyObject *>::unpack_tuple(args,
			kwargs,
			"count",
			std::integral_constant<size_t, 0>{},
			std::integral_constant<size_t, 2>{},
			nullptr,
			nullptr);
		if (parsed_args.is_err()) { return Err(parsed_args.unwrap_err()); }

		auto [start, step] = parsed_args.unwrap();

		if ((!start || start == py_none()) && (!step || step == py_none())) {
			return Count::create();
		}

		Number start_{ 0 };
		Number step_{ 1 };

		if (start && start != py_none()) {
			if (start->type()->issubclass(types::integer())) {
				start_ = static_cast<const PyInteger &>(*start).value();
			} else if (start->type()->issubclass(types::float_())) {
				start_ = static_cast<const PyFloat &>(*start).value();
			} else {
				// TODO also handle complex
				return Err(type_error("a number is required"));
			}
		}

		if (step && step != py_none()) {
			if (step->type()->issubclass(types::integer())) {
				step_ = static_cast<const PyInteger &>(*step).value();
			} else if (step->type()->issubclass(types::float_())) {
				step_ = static_cast<const PyFloat &>(*step).value();
			} else {
				// TODO also handle complex
				return Err(type_error("a number is required"));
			}
		}

		return Count::create(std::move(start_), std::move(step_));
	}

	PyResult<PyObject *> Count::__iter__() const { return Ok(const_cast<Count *>(this)); }

	PyResult<PyObject *> Count::__next__()
	{
		auto to_return = m_current;

		m_current += m_step;

		return PyObject::from(to_return);
	}

	void Count::visit_graph(Visitor &visitor) { PyObject::visit_graph(visitor); }

	PyType *Count::register_type(PyModule *module)
	{
		if (!s_itertools_count) {
			s_itertools_count = klass<Count>(module, "itertools.count").finalize();
		}
		return s_itertools_count;
	}

}// namespace itertools
}// namespace py
