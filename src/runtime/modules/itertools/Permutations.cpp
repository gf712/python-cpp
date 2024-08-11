#include "Permutations.hpp"
#include "runtime/MemoryError.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/Value.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>

namespace py {
namespace {
	static PyType *s_itertools_permutations = nullptr;
}
namespace itertools {

	Permutations::Permutations(PyType *type) : PyBaseObject(type) {}

	Permutations::Permutations(PyList *pool, size_t length)
		: PyBaseObject(s_itertools_permutations), m_pool(pool), m_length(length),
		  m_iterator_length(m_pool->elements().size()), m_inner_iteration(m_length - 1),
		  m_indices(m_iterator_length), m_done(m_length > m_iterator_length)
	{
		std::iota(m_indices.begin(), m_indices.end(), 0);
		if (!m_done) {
			m_cycles.reserve(m_length);
			for (size_t i = m_iterator_length; i > (m_iterator_length - m_length); --i) {
				m_cycles.push_back(i);
			}
		}
	}

	PyResult<PyObject *> Permutations::create(PyObject *iterable, std::optional<size_t> length)
	{
		return iterable->iter().and_then([length](PyObject *iterator) -> PyResult<PyObject *> {
			auto pool_ = PyList::create();
			if (pool_.is_err()) { return pool_; }
			auto *pool = pool_.unwrap();

			auto value_ = iterator->next();
			while (value_.is_ok()) {
				pool->elements().push_back(value_.unwrap());
				value_ = iterator->next();
			}

			if (!value_.unwrap_err()->type()->issubclass(types::stop_iteration())) {
				return value_;
			}


			auto *obj = VirtualMachine::the().heap().allocate<Permutations>(
				pool, length.value_or(pool->elements().size()));
			if (!obj) { return Err(memory_error(sizeof(Permutations))); }
			return Ok(obj);
		});
	}

	PyResult<PyObject *> Permutations::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
	{
		ASSERT(s_itertools_permutations);
		ASSERT(type == s_itertools_permutations);

		auto parsed_args = PyArgsParser<PyObject *, PyInteger *>::unpack_tuple(args,
			kwargs,
			"itertools.iterator",
			std::integral_constant<size_t, 1>{},
			std::integral_constant<size_t, 2>{},
			nullptr);
		if (parsed_args.is_err()) { return Err(parsed_args.unwrap_err()); }

		auto [iterable, length] = parsed_args.unwrap();

		std::optional<size_t> length_;
		if (length) { length_ = length->as_size_t(); }

		return Permutations::create(iterable, std::move(length_));
	}

	PyResult<PyObject *> Permutations::__iter__() const
	{
		return Ok(const_cast<Permutations *>(this));
	}

	PyResult<PyObject *> Permutations::__next__()
	{
		if (m_done) { return Err(stop_iteration()); }
		if (m_first) {
			m_first = false;
			std::vector<Value> result;
			result.reserve(m_length);
			for (size_t i = 0; i < m_length; ++i) { result.push_back(m_pool->elements()[i]); }
			return PyTuple::create(std::move(result));
		}

		m_cycles[m_inner_iteration] -= 1;
		while (m_cycles[m_inner_iteration] == 0) {
			std::rotate(m_indices.begin() + m_inner_iteration,
				m_indices.begin() + m_inner_iteration + 1,
				m_indices.end());
			m_cycles[m_inner_iteration] = m_iterator_length - m_inner_iteration;

			if (m_inner_iteration == 0) {
				m_done = true;
				return Err(stop_iteration());
			} else {
				--m_inner_iteration;
			}
			m_cycles[m_inner_iteration] -= 1;
		}

		const auto j = m_indices.size() - m_cycles[m_inner_iteration];
		std::iter_swap(m_indices.begin() + j, m_indices.begin() + m_inner_iteration);
		std::vector<Value> result;
		result.reserve(m_length);
		for (size_t i = 0; i < m_length; ++i) {
			result.push_back(m_pool->elements()[m_indices[i]]);
		}
		m_inner_iteration = m_length - 1;
		return PyTuple::create(std::move(result));
	}

	PyType *Permutations::register_type(PyModule *module)
	{
		if (!s_itertools_permutations) {
			s_itertools_permutations = klass<Permutations>(module, "itertools.starmap").finalize();
		}
		return s_itertools_permutations;
	}

	void Permutations::visit_graph(Visitor &visitor)
	{
		PyObject::visit_graph(visitor);
		if (m_pool) { visitor.visit(*m_pool); }
	}
}// namespace itertools
}// namespace py
