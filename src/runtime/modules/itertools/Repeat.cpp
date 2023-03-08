#include "Repeat.hpp"
#include "runtime/MemoryError.hpp"
#include "runtime/PyArgParser.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/types/api.hpp"

namespace py {
namespace {
	static PyType *s_itertools_repeat = nullptr;
}
namespace itertools {
	Repeat::Repeat(PyType *type) : PyBaseObject(type) {}

	Repeat::Repeat(PyObject *object, BigIntType times)
		: PyBaseObject(s_itertools_repeat), m_object(object), m_times_remaining(std::move(times))
	{}

	Repeat::Repeat(PyObject *object) : PyBaseObject(s_itertools_repeat), m_object(object) {}

	PyResult<PyObject *> Repeat::create(PyObject *object, BigIntType times)
	{
		if (times < 0) { times = 0; }
		auto *obj = VirtualMachine::the().heap().allocate<Repeat>(object, times);
		if (!obj) { return Err(memory_error(sizeof(Repeat))); }
		return Ok(obj);
	}

	PyResult<PyObject *> Repeat::create(PyObject *object)
	{
		auto *obj = VirtualMachine::the().heap().allocate<Repeat>(object);
		if (!obj) { return Err(memory_error(sizeof(Repeat))); }
		return Ok(obj);
	}

	PyResult<PyObject *> Repeat::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
	{
		ASSERT(s_itertools_repeat);
		ASSERT(type == s_itertools_repeat);

		auto parsed_args = PyArgsParser<PyObject *, PyObject *>::unpack_tuple(args,
			kwargs,
			"repeat",
			std::integral_constant<size_t, 1>{},
			std::integral_constant<size_t, 2>{},
			py_none());
		if (parsed_args.is_err()) { return Err(parsed_args.unwrap_err()); }

		auto [obj, times] = parsed_args.unwrap();
		if (times == py_none()) { return Repeat::create(obj); }

		if (!as<PyInteger>(times)) {
			return Err(type_error("integer argument expected, got {}", times->type()->name()));
		}

		return Repeat::create(obj, as<PyInteger>(times)->as_big_int());
	}

	PyResult<PyObject *> Repeat::__iter__() const { return Ok(const_cast<Repeat *>(this)); }

	PyResult<PyObject *> Repeat::__next__()
	{
		ASSERT(m_times_remaining.value_or(0) >= 0);
		if (m_times_remaining.has_value() && m_times_remaining == 0) {
			return Err(stop_iteration());
		}
		if (m_times_remaining.has_value()) { --(*m_times_remaining); };

		return Ok(m_object);
	}

	void Repeat::visit_graph(Visitor &visitor)
	{
		PyObject::visit_graph(visitor);
		if (m_object) { visitor.visit(*m_object); }
	}

	PyType *Repeat::register_type(PyModule *module)
	{
		if (!s_itertools_repeat) {
			s_itertools_repeat = klass<Repeat>(module, "itertools.repeat").finalize();
		}
		return s_itertools_repeat;
	}

}// namespace itertools
}// namespace py
