#include "Product.hpp"
#include "runtime/MemoryError.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/Value.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"
#include <variant>

namespace py {
namespace {
	static PyType *s_itertools_product = nullptr;
}
namespace itertools {

	Product::Product(PyType *type) : PyBaseObject(type) {}

	Product::Product(PyList *pools, size_t repeat)
		: PyBaseObject(s_itertools_product), m_pools(pools), m_repeat(repeat)
	{}

	PyResult<PyObject *> Product::create(PyObject *iterable, std::optional<size_t> repeat)
	{
		if (repeat.has_value() && repeat == 0) {
			auto *obj = VirtualMachine::the().heap().allocate<Product>(
				PyList::create().unwrap(), size_t{ 0 });
			if (!obj) { return Err(memory_error(sizeof(Product))); }
			return Ok(obj);
		}
		return iterable->iter().and_then([repeat](PyObject *iterator) -> PyResult<PyObject *> {
			auto pools_ = PyList::create();
			if (pools_.is_err()) { return pools_; }
			auto *pools = pools_.unwrap();
			auto value_ = iterator->next();
			while (value_.is_ok()) {
				auto pool = value_.unwrap()->iter().and_then(
					[](PyObject *iterator) -> PyResult<PyObject *> {
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
						return Ok(pool);
					});
				if (pool.is_err()) { return pool; }
				pools->elements().push_back(pool.unwrap());
				value_ = iterator->next();
			}
			const size_t pool_size = pools->elements().size();
			for (size_t i = 0; i < repeat.value_or(1) - 1; ++i) {
				auto &els = pools->elements();
				els.insert(els.end(), els.begin(), els.begin() + pool_size);
			}
			auto *obj = VirtualMachine::the().heap().allocate<Product>(pools, repeat.value_or(1));
			if (!obj) { return Err(memory_error(sizeof(Product))); }
			return Ok(obj);
		});
	}

	PyResult<PyObject *> Product::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
	{
		ASSERT(s_itertools_product);
		ASSERT(type == s_itertools_product);

		std::optional<size_t> length_;
		if (kwargs) {
			if (auto it = kwargs->map().find(String{ "repeat" }); it != kwargs->map().end()) {
				auto obj = PyObject::from(it.value());
				if (obj.is_err()) { return obj; }
				if (!obj.unwrap()->type()->issubclass(types::integer())) {
					return Err(type_error("'{}' object cannot be interpreted as an integer",
						obj.unwrap()->type()->name()));
				}
				auto l = static_cast<const PyInteger &>(*obj.unwrap()).as_big_int();
				if (l < 0) {
					length_ = 0;
				} else {
					length_ = l.get_ui();
				}
			}
		}

		return Product::create(args, std::move(length_));
	}

	PyResult<PyObject *> Product::__iter__() const { return Ok(const_cast<Product *>(this)); }

	PyResult<PyObject *> Product::__next__()
	{
		if (m_repeat == 0) { return Err(stop_iteration()); }

		if (m_result.empty()) {
			m_result.emplace_back();
			for (const auto &pool : m_pools->elements()) {
				ASSERT(std::holds_alternative<PyObject *>(pool));
				ASSERT(as<PyList>(std::get<PyObject *>(pool)));
				std::vector<std::vector<Value>> result;
				auto *pool_list = as<PyList>(std::get<PyObject *>(pool));
				for (auto x : m_result) {
					for (auto y : pool_list->elements()) {
						auto tmp = x;
						tmp.push_back(y);
						result.push_back(std::move(tmp));
					}
				}
				m_result = std::move(result);
			}
		}

		if (m_iteration_count < m_result.size()) {
			return PyTuple::create(std::move(m_result[m_iteration_count++]));
		}

		return Err(stop_iteration());
	}

	PyType *Product::register_type(PyModule *module)
	{
		if (!s_itertools_product) {
			s_itertools_product = klass<Product>(module, "itertools.product").finalize();
		}
		return s_itertools_product;
	}

	void Product::visit_graph(Visitor &visitor)
	{
		PyObject::visit_graph(visitor);
		if (m_pools) { visitor.visit(*m_pools); }
		for (auto &vec : m_result) {
			for (auto &el : vec) {
				if (std::holds_alternative<PyObject *>(el)) {
					auto obj = std::get<PyObject *>(el);
					if (obj) { visitor.visit(*obj); }
				}
			}
		}
	}
}// namespace itertools
}// namespace py
