#include "ISlice.hpp"
#include "runtime/MemoryError.hpp"
#include "runtime/PyArgParser.hpp"
#include "runtime/PyType.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/ValueError.hpp"
#include "runtime/types/api.hpp"

namespace py {
namespace {
	PyType *s_itertools_islice = nullptr;
}

namespace itertools {
	ISlice::ISlice(PyType *type) : PyBaseObject(type) {}

	ISlice::ISlice(PyObject *iterator,
		BigIntType start,
		std::optional<BigIntType> stop,
		BigIntType step)
		: PyBaseObject(s_itertools_islice), m_iterator(iterator), m_start(std::move(start)),
		  m_stop(std::move(stop)), m_step(std::move(step))
	{}

	PyResult<PyObject *>
		ISlice::create(PyObject *iterable, PyObject *start, PyObject *stop, PyObject *step)
	{
		auto get_big_int = [](PyObject *parameter,
							   std::string_view name) -> PyResult<std::optional<BigIntType>> {
			if (parameter == py_none()) {
				return Ok(std::nullopt);
			} else if (auto n = as<PyInteger>(parameter)) {
				if (n->as_big_int() < 0) {
					return Err(value_error(
						"{} argument for islice() an integer: 0 <= x <=  sys.maxsize", name));
				}
				return Ok(n->as_big_int());
			} else {
				return Err(
					value_error("{} argument for islice() must be None or an integer: 0 <= x <= "
								"sys.maxsize",
						name));
			}
		};

		auto iterator = iterable->iter();
		if (iterator.is_err()) { return iterator; }

		auto start_ = get_big_int(start, "start");
		if (start_.is_err()) { return Err(start_.unwrap_err()); }
		auto stop_ = get_big_int(stop, "stop");
		if (stop_.is_err()) { return Err(stop_.unwrap_err()); }
		auto step_ = get_big_int(step, "step");
		if (step_.is_err()) { return Err(step_.unwrap_err()); }

		auto *obj = VirtualMachine::the().heap().allocate<ISlice>(iterator.unwrap(),
			start_.unwrap().value_or(BigIntType{ 0 }),
			stop_.unwrap(),
			step_.unwrap().value_or(BigIntType{ 1 }));
		if (!obj) { return Err(memory_error(sizeof(ISlice))); }
		return Ok(obj);
	}

	PyResult<PyObject *> ISlice::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
	{
		ASSERT(s_itertools_islice);
		ASSERT(type == s_itertools_islice);

		PyObject *iterable = nullptr;
		PyObject *start = py_none();
		PyObject *step = py_none();
		PyObject *stop = py_none();

		if (args && args->size() == 2) {
			auto parsed_args = PyArgsParser<PyObject *, PyObject *>::unpack_tuple(args,
				kwargs,
				"islice",
				std::integral_constant<size_t, 2>{},
				std::integral_constant<size_t, 2>{});
			if (parsed_args.is_err()) { return Err(parsed_args.unwrap_err()); }
			iterable = std::get<0>(parsed_args.unwrap());
			stop = std::get<1>(parsed_args.unwrap());
		} else {
			auto parsed_args =
				PyArgsParser<PyObject *, PyObject *, PyObject *, PyObject *>::unpack_tuple(args,
					kwargs,
					"islice",
					std::integral_constant<size_t, 3>{},
					std::integral_constant<size_t, 4>{},
					py_none());
			if (parsed_args.is_err()) { return Err(parsed_args.unwrap_err()); }
			iterable = std::get<0>(parsed_args.unwrap());
			start = std::get<1>(parsed_args.unwrap());
			stop = std::get<2>(parsed_args.unwrap());
			step = std::get<3>(parsed_args.unwrap());
		}

		return ISlice::create(iterable, start, stop, step);
	}

	PyResult<PyObject *> ISlice::__iter__() const { return Ok(const_cast<ISlice *>(this)); }

	PyResult<PyObject *> ISlice::__next__()
	{
		auto advance_iterator_to = [this](const BigIntType &to) -> PyResult<std::monostate> {
			while (m_counter < to) {
				auto result = m_iterator->next();
				if (result.is_err()) { return Err(result.unwrap_err()); }
				(*m_counter)++;
			}
			return Ok(std::monostate{});
		};
		if (!m_counter.has_value()) {
			m_counter = 0;
			if (const auto result = advance_iterator_to(m_start); result.is_err()) {
				return Err(result.unwrap_err());
			}
		} else {
			const BigIntType next = (*m_counter) + m_step - 1;
			if (const auto result = advance_iterator_to(next); result.is_err()) {
				return Err(result.unwrap_err());
			}
		}

		if (m_stop.has_value() && m_counter >= *m_stop) { return Err(stop_iteration()); }

		auto result = m_iterator->next();
		(*m_counter)++;
		if (result.is_err()) { return result; }

		return result;
	}

	PyType *ISlice::register_type(PyModule *module)
	{
		if (!s_itertools_islice) {
			s_itertools_islice = klass<ISlice>(module, "itertools.islice").finalize();
		}
		return s_itertools_islice;
	}

	void ISlice::visit_graph(Visitor &visitor)
	{
		PyObject::visit_graph(visitor);
		if (m_iterator) { visitor.visit(*m_iterator); }
	}

}// namespace itertools
}// namespace py
