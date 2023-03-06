#include "Chain.hpp"
#include "runtime/MemoryError.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/types/api.hpp"

namespace py {
namespace {
	static PyType *s_itertools_chain = nullptr;
}
namespace itertools {

	Chain::Chain(PyObject *iterable_objects_iterator)
		: PyBaseObject(s_itertools_chain), m_iterable_objects_iterator(iterable_objects_iterator)
	{}

	PyResult<PyObject *> Chain::create(PyTuple *iterable_objects)
	{
		auto iterable_objects_iterator = PyTupleIterator::create(*iterable_objects);
		if (iterable_objects_iterator.is_err()) {
			return Err(iterable_objects_iterator.unwrap_err());
		}
		auto *obj =
			VirtualMachine::the().heap().allocate<Chain>(iterable_objects_iterator.unwrap());
		if (!obj) { return Err(memory_error(sizeof(Chain))); }
		return Ok(obj);
	}

	PyResult<PyObject *> Chain::create(PyObject *iterable_objects)
	{
		auto iterable_objects_iterator = iterable_objects->iter();
		if (iterable_objects_iterator.is_err()) {
			return Err(iterable_objects_iterator.unwrap_err());
		}
		auto *obj =
			VirtualMachine::the().heap().allocate<Chain>(iterable_objects_iterator.unwrap());
		if (!obj) { return Err(memory_error(sizeof(Chain))); }
		return Ok(obj);
	}

	PyResult<PyObject *> Chain::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
	{
		ASSERT(s_itertools_chain);
		ASSERT(type == s_itertools_chain);
		if (kwargs && kwargs->size() > 0) {
			return Err(type_error("chain() takes no keyword arguments"));
		}

		return Chain::create(args);
	}

	PyResult<PyObject *> Chain::__iter__() const { return Ok(const_cast<Chain *>(this)); }

	PyResult<PyObject *> Chain::__next__()
	{
		auto get_next_iterator = [this]() -> PyResult<PyObject *> {
			if (auto current_iterator = m_iterable_objects_iterator->next();
				current_iterator.is_ok()) {
				m_current_iterator = current_iterator.unwrap();
			} else {
				return current_iterator;
			}
			return m_current_iterator->iter();
		};
		if (!m_current_iterator) {
			if (auto current_iterator = get_next_iterator(); current_iterator.is_ok()) {
				m_current_iterator = current_iterator.unwrap();
			} else {
				return current_iterator;
			}
		}

		ASSERT(m_current_iterator);
		auto next = m_current_iterator->next();
		while (next.is_err() && next.unwrap_err()->type()->issubclass(stop_iteration()->type())) {
			if (auto current_iterator = get_next_iterator(); current_iterator.is_ok()) {
				m_current_iterator = current_iterator.unwrap();
			} else {
				return current_iterator;
			}
			next = m_current_iterator->next();
		}

		return next;
	}

	PyResult<PyObject *> Chain::from_iterable(PyType *type, PyTuple *args, PyDict *kwargs)
	{
		ASSERT(s_itertools_chain);
		ASSERT(type == s_itertools_chain);

		auto parsed_args = PyArgsParser<PyObject *>::unpack_tuple(args,
			kwargs,
			"chain.from_itertools",
			std::integral_constant<size_t, 1>{},
			std::integral_constant<size_t, 1>{});
		if (parsed_args.is_err()) { return Err(parsed_args.unwrap_err()); }
		auto [iterable] = parsed_args.unwrap();
		return Chain::create(iterable);
	}

	void Chain::visit_graph(Visitor &visitor)
	{
		PyObject::visit_graph(visitor);
		if (m_iterable_objects_iterator) { visitor.visit(*m_iterable_objects_iterator); }
		if (m_current_iterator) { visitor.visit(*m_current_iterator); }
	}

	PyType *Chain::register_type(PyModule *module)
	{
		if (!s_itertools_chain) {
			s_itertools_chain = klass<Chain>(module, "itertools.chain")
									.classmethod("from_iterable", &Chain::from_iterable)
									.finalize();
		}
		return s_itertools_chain;
	}
}// namespace itertools
}// namespace py
