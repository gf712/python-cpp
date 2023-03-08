#include "StarMap.hpp"
#include "runtime/MemoryError.hpp"
#include "runtime/PyList.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/types/api.hpp"

namespace py {
namespace {
	static PyType *s_itertools_starmap = nullptr;
}
namespace itertools {

	StarMap::StarMap(PyType *type) : PyBaseObject(type) {}

	StarMap::StarMap(PyObject *function, PyObject *iterator)
		: PyBaseObject(s_itertools_starmap), m_function(function), m_iterator(iterator)
	{}

	PyResult<PyObject *> StarMap::create(PyObject *function, PyObject *iterable)
	{
		return iterable->iter().and_then([function](PyObject *iterator) -> PyResult<PyObject *> {
			auto *obj = VirtualMachine::the().heap().allocate<StarMap>(function, iterator);
			if (!obj) { return Err(memory_error(sizeof(StarMap))); }
			return Ok(obj);
		});
	}

	PyResult<PyObject *> StarMap::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
	{
		ASSERT(s_itertools_starmap);
		ASSERT(type == s_itertools_starmap);

		auto parsed_args = PyArgsParser<PyObject *, PyObject *>::unpack_tuple(args,
			kwargs,
			"starmap",
			std::integral_constant<size_t, 2>{},
			std::integral_constant<size_t, 2>{});
		if (parsed_args.is_err()) { return Err(parsed_args.unwrap_err()); }

		auto [func, iterable] = parsed_args.unwrap();

		return StarMap::create(func, iterable);
	}

	PyResult<PyObject *> StarMap::__iter__() const { return Ok(const_cast<StarMap *>(this)); }

	PyResult<PyObject *> StarMap::__next__()
	{
		auto current_args_iterable_ = m_iterator->next();
		if (current_args_iterable_.is_err()) { return current_args_iterable_; }

		auto *current_args_iterable = current_args_iterable_.unwrap();
		auto current_args_iterator = current_args_iterable->iter();
		if (current_args_iterator.is_err()) { return current_args_iterator; }

		auto args_ = PyList::create();
		if (args_.is_err()) { return args_; }
		auto *args = args_.unwrap();
		auto next = current_args_iterator.unwrap()->next();
		while (next.is_ok()) {
			args->elements().push_back(next.unwrap());
			next = current_args_iterator.unwrap()->next();
		}

		if (next.unwrap_err()->type() != stop_iteration()->type()) { return next; }

		auto args_tuple = PyTuple::create(args->elements());
		if (args_tuple.is_err()) { return args_tuple; }
		return m_function->call(args_tuple.unwrap(), nullptr);
	}

	PyType *StarMap::register_type(PyModule *module)
	{
		if (!s_itertools_starmap) {
			s_itertools_starmap = klass<StarMap>(module, "itertools.starmap").finalize();
		}
		return s_itertools_starmap;
	}

	void StarMap::visit_graph(Visitor &visitor)
	{
		PyObject::visit_graph(visitor);
		if (m_function) { visitor.visit(*m_function); }
		if (m_iterator) { visitor.visit(*m_iterator); }
	}
}// namespace itertools
}// namespace py
