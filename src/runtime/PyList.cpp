#include "PyList.hpp"
#include "IndexError.hpp"
#include "MemoryError.hpp"
#include "PyBool.hpp"
#include "PyDict.hpp"
#include "PyFunction.hpp"
#include "PyGenericAlias.hpp"
#include "PyInteger.hpp"
#include "PyNone.hpp"
#include "PyNumber.hpp"
#include "PySlice.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "StopIteration.hpp"
#include "ValueError.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/Value.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <unordered_set>
#include <variant>


namespace py {

static std::unordered_set<PyObject *> visited;

template<> PyList *as(PyObject *obj)
{
	if (obj->type() == types::list()) { return static_cast<PyList *>(obj); }
	return nullptr;
}

template<> const PyList *as(const PyObject *obj)
{
	if (obj->type() == types::list()) { return static_cast<const PyList *>(obj); }
	return nullptr;
}

PyList::PyList() : PyBaseObject(types::BuiltinTypes::the().list()) {}

PyList::PyList(PyType *type) : PyBaseObject(type) {}

PyList::PyList(std::vector<Value> elements) : PyList() { m_elements = std::move(elements); }

PyResult<PyList *> PyList::create(std::vector<Value> elements)
{
	auto *result = VirtualMachine::the().heap().allocate<PyList>(std::move(elements));
	if (!result) { return Err(memory_error(sizeof(PyList))); }
	return Ok(result);
}

PyResult<PyList *> PyList::create(std::span<const Value> s)
{
	std::vector<Value> elements{ s.size(), nullptr };
	for (size_t idx = 0; auto el : s) { elements[idx++] = el; }

	auto *result = VirtualMachine::the().heap().allocate<PyList>(std::move(elements));
	if (!result) { return Err(memory_error(sizeof(PyList))); }
	return Ok(result);
}

PyResult<PyList *> PyList::create()
{
	auto *result = VirtualMachine::the().heap().allocate<PyList>();
	if (!result) { return Err(memory_error(sizeof(PyList))); }
	return Ok(result);
}

PyResult<PyObject *> PyList::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	if (!type->issubclass(types::list())) {
		return Err(type_error(
			"list.__new__({}): {} is not a subtype of list", type->name(), type->name()));
	}

	auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
		kwargs,
		"list",
		std::integral_constant<size_t, 0>{},
		std::integral_constant<size_t, 1>{},
		nullptr /* iterable */);

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [iterable] = result.unwrap();
	if (!iterable) { return PyList::create(); }

	auto iterator_ = iterable->iter();
	if (iterator_.is_err()) { return iterator_; }
	auto iterator = iterator_.unwrap();

	auto els_ = PyList::create();
	if (els_.is_err()) { return Err(els_.unwrap_err()); }
	auto els = els_.unwrap();

	auto value = iterator->next();
	while (value.is_ok()) {
		els->elements().push_back(value.unwrap());
		value = iterator->next();
	}

	if (!value.unwrap_err()->type()->issubclass(stop_iteration()->type())) { return value; }

	return Ok(els);
}

PyResult<PyObject *> PyList::append(PyObject *element)
{
	m_elements.push_back(element);
	return Ok(py_none());
}

PyResult<PyObject *> PyList::extend(PyObject *iterable)
{
	auto iterator = iterable->iter();
	if (iterator.is_err()) return iterator;

	auto tmp_list_ = PyList::create();
	if (tmp_list_.is_err()) return tmp_list_;
	auto *tmp_list = tmp_list_.unwrap();
	auto value = iterator.unwrap()->next();
	while (value.is_ok()) {
		tmp_list->append(value.unwrap());
		value = iterator.unwrap()->next();
	}

	if (!value.unwrap_err()->type()->issubclass(stop_iteration()->type())) { return value; }

	m_elements.insert(m_elements.end(), tmp_list->elements().begin(), tmp_list->elements().end());

	return Ok(py_none());
}

PyResult<PyObject *> PyList::pop(PyObject *index)
{
	if (m_elements.empty()) { return Err(index_error("pop from empty list")); }

	if (index) {
		if (!as<PyInteger>(index)) {
			return Err(type_error(
				"'{}' object cannot be interpreted as an integer", index->type()->name()));
		}
		auto idx = [index, this]() -> PyResult<size_t> {
			auto idx_value = as<PyInteger>(index)->as_i64();
			size_t idx = m_elements.size();
			if (idx_value < 0) {
				if (static_cast<uint64_t>(std::abs(idx_value)) > m_elements.size()) {
					return Err(index_error("pop index '{}' out of range for list of size '{}'",
						idx,
						m_elements.size()));
				}
				idx += idx_value;
			} else {
				idx = static_cast<size_t>(idx_value);
			}
			if (idx >= m_elements.size()) {
				return Err(index_error(
					"pop index '{}' out of range for list of size '{}'", idx, m_elements.size()));
			}
			return Ok(idx);
		}();
		return idx.and_then([this](size_t idx) {
			return PyObject::from(m_elements[idx]).and_then([this, idx](PyObject *el) {
				if (idx == m_elements.size()) {
					m_elements.pop_back();
				} else {
					m_elements.erase(m_elements.begin() + idx);
				}
				return Ok(el);
			});
		});
	} else {
		return PyObject::from(m_elements.back()).and_then([this](PyObject *el) {
			m_elements.pop_back();
			return Ok(el);
		});
	}
}

std::string PyList::to_string() const
{
	auto r = __repr__();
	if (r.is_err()) { return "<list to string error>"; }
	return as<PyString>(r.unwrap())->to_string();
}

PyResult<PyObject *> PyList::__repr__() const
{
	std::ostringstream os;

	[[maybe_unused]] struct Cleanup
	{
		const PyList *list;
		bool do_cleanup;

		~Cleanup()
		{
			if (do_cleanup) {
				auto it = visited.find(const_cast<PyList *>(list));
				if (it != visited.end()) { visited.erase(it); }
			}
		}
	} cleanup{ this, !visited.contains(const_cast<PyList *>(this)) };
	visited.insert(const_cast<PyList *>(this));

	auto repr = [](const auto &el) -> PyResult<PyString *> {
		return std::visit(overloaded{
							  [](const auto &value) { return PyString::create(value.to_string()); },
							  [](PyObject *value) {
								  if (visited.contains(value)) { return PyString::create("[...]"); }
								  return value->repr();
							  },
						  },
			el);
	};
	os << "[";
	if (!m_elements.empty()) {
		auto it = m_elements.begin();
		while (std::next(it) != m_elements.end()) {
			auto r = repr(*it);
			if (r.is_err()) { return r; }
			os << std::move(r.unwrap()->value()) << ", ";
			std::advance(it, 1);
		}
		auto r = repr(*it);
		if (r.is_err()) { return r; }
		os << std::move(r.unwrap()->value());
	}
	os << "]";

	return PyString::create(os.str());
}

PyResult<PyObject *> PyList::__iter__() const
{
	auto &heap = VirtualMachine::the().heap();
	auto *it = heap.allocate<PyListIterator>(*this);
	if (!it) { return Err(memory_error(sizeof(PyListIterator))); }
	return Ok(it);
}

PyResult<PyObject *> PyList::__getitem__(int64_t index)
{
	if (index < 0) {
		if (static_cast<size_t>(std::abs(index)) > m_elements.size()) {
			return Err(index_error("list index out of range"));
		}
		index += m_elements.size();
	}
	ASSERT(index >= 0);
	if (static_cast<size_t>(index) >= m_elements.size()) {
		return Err(index_error("list index out of range"));
	}
	return PyObject::from(m_elements[index]);
}

PyResult<std::monostate> PyList::__setitem__(int64_t index, PyObject *value)
{
	if (index < 0) { index += m_elements.size(); }
	if (static_cast<size_t>(index) >= m_elements.size()) {
		return Err(index_error("list index out of range"));
	}
	m_elements[index] = value;
	return Ok(std::monostate{});
}

PyResult<std::monostate> PyList::__delitem__(PyObject *key)
{
	if (!key->type()->issubclass(types::integer()) && !key->type()->issubclass(types::slice())) {
		return Err(type_error(
			"list indices must be integers or slices, not {}", key->type()->to_string()));
	}

	auto validate_index = [this](BigIntType index_value) -> PyResult<size_t> {
		if (index_value >= 0) {
			ASSERT(index_value.fits_ulong_p());
			const auto index = index_value.get_ui();
			if (index > m_elements.size()) {
				return Err(index_error("list deletion index out of range"));
			}
			return Ok(index);
		} else {
			ASSERT(index_value.fits_slong_p());
			const auto index = index_value.get_si();
			if (static_cast<size_t>(std::abs(index)) > m_elements.size()) {
				return Err(index_error("list deletion index out of range"));
			}
			return Ok(m_elements.size() - std::abs(index));
		}
		ASSERT_NOT_REACHED();
	};

	auto delete_index = [this, validate_index](BigIntType index_value) -> PyResult<std::monostate> {
		const auto index = validate_index(index_value);
		if (index.is_err()) { return Err(index.unwrap_err()); }
		m_elements.erase(m_elements.begin() + index.unwrap());
		return Ok(std::monostate{});
	};

	if (key->type()->issubclass(types::slice())) {
		const auto *slice = static_cast<const PySlice *>(key);
		auto unpack_indices = slice->unpack();
		if (unpack_indices.is_err()) { return Err(unpack_indices.unwrap_err()); }
		auto [start, stop, step] = unpack_indices.unwrap();
		start = start == std::numeric_limits<int64_t>::max()
					? static_cast<int64_t>(m_elements.size()) - 1
					: start;
		stop = stop == std::numeric_limits<int64_t>::min()
				   ? static_cast<int64_t>(m_elements.size()) - 1
				   : start;
		if (step == 0) { return Err(value_error("slice step cannot be zero")); }
		auto start_index = validate_index(start);
		if (start_index.is_err()) { return Err(start_index.unwrap_err()); }
		auto stop_index = validate_index(stop);
		if (stop_index.is_err()) { return Err(stop_index.unwrap_err()); }
		start = start_index.unwrap();
		stop = stop_index.unwrap();
		if (step > 0) {
			if (start > stop) { return Ok(std::monostate{}); }
			if (step == 1) {
				m_elements.erase(m_elements.begin() + start, m_elements.begin() + stop);
			} else {
				for (auto idx = start; idx < stop; idx += step) {
					auto result = delete_index(idx);
					if (result.is_err()) { return result; }
					idx -= 1;
					stop -= 1;
				}
			}
		} else if (step < 0) {
			if (stop >= start) { return Ok(std::monostate{}); }
			for (auto idx = start - 1; idx > stop; idx += step) {
				auto result = delete_index(idx);
				if (result.is_err()) { return result; }
			}
		}
	} else {
		ASSERT(key->type()->issubclass(types::integer()));
		const auto index_value = static_cast<PyInteger &>(*key).as_big_int();
		return delete_index(index_value);
	}

	return Ok(std::monostate{});
}

PyResult<PyObject *> PyList::__getitem__(PyObject *index)
{
	if (auto index_int = as<PyInteger>(index)) {
		const auto i = index_int->as_i64();
		return __getitem__(i);
	} else if (auto slice = as<PySlice>(index)) {
		auto indices_ = slice->unpack();
		if (indices_.is_err()) return Err(indices_.unwrap_err());
		const auto [start_, end_, step] = indices_.unwrap();

		const auto [start, end, slice_length] =
			PySlice::adjust_indices(start_, end_, step, m_elements.size());

		if (slice_length == 0) { return PyList::create(); }
		if (start == 0 && end == static_cast<int64_t>(m_elements.size()) && step == 1) {
			// shallow copy of the list since we need all elements
			return PyList::create(m_elements);
		}

		auto new_list = PyList::create();
		if (new_list.is_err()) return new_list;

		for (int64_t idx = start, i = 0; i < slice_length; idx += step, ++i) {
			new_list.unwrap()->elements().push_back(m_elements[idx]);
		}
		return new_list;
	} else {
		return Err(
			type_error("list indices must be integers or slices, not {}", index->type()->name()));
	}
}

PyResult<std::monostate> PyList::__setitem__(PyObject *index, PyObject *value)
{
	if (index->type()->issubclass(types::integer())) {
		const auto i = static_cast<const PyInteger &>(*index).as_i64();
		return __setitem__(i, value);
	} else if (index->type()->issubclass(types::slice())) {
		auto value_iter = value->iter();
		if (value_iter.is_err()) { return Err(type_error("can only assign an iterable")); }
		auto validate_index = [this](BigIntType index_value) -> PyResult<size_t> {
			if (index_value >= 0) {
				ASSERT(index_value.fits_ulong_p());
				const auto index = index_value.get_ui();
				if (index > m_elements.size()) {
					return Err(index_error("list assignment index out of range"));
				}
				return Ok(index);
			} else {
				ASSERT(index_value.fits_slong_p());
				const auto index = index_value.get_si();
				if (static_cast<size_t>(std::abs(index)) > m_elements.size()) {
					return Err(index_error("list assignment index out of range"));
				}
				return Ok(m_elements.size() - std::abs(index));
			}
			ASSERT_NOT_REACHED();
		};
		const auto &slice = static_cast<const PySlice &>(*index);
		auto unpack_indices = slice.unpack();
		if (unpack_indices.is_err()) { return Err(unpack_indices.unwrap_err()); }
		auto [start, stop, step] = unpack_indices.unwrap();
		start = start == std::numeric_limits<int64_t>::max()
					? static_cast<int64_t>(m_elements.size()) - 1
					: start;
		stop = stop == std::numeric_limits<int64_t>::min()
				   ? static_cast<int64_t>(m_elements.size()) - 1
				   : start;
		if (step == 0) { return Err(value_error("slice step cannot be zero")); }
		if (step != 1) { TODO(); }
		auto start_index = validate_index(start);
		if (start_index.is_err()) { return Err(start_index.unwrap_err()); }
		auto stop_index = validate_index(stop);
		if (stop_index.is_err()) { return Err(stop_index.unwrap_err()); }
		start = start_index.unwrap();
		stop = stop_index.unwrap();

		auto val = value_iter.unwrap()->next();
		auto i = start;
		for (; i < stop && val.is_ok(); i += step) {
			auto index_ = validate_index(i);
			if (index_.is_err()) { return Err(index_.unwrap_err()); }
			m_elements[index_.unwrap()] = val.unwrap();
			val = value_iter.unwrap()->next();
		}
		while (val.is_ok()) {
			m_elements.insert(m_elements.begin() + i, val.unwrap());
			val = value_iter.unwrap()->next();
			++i;
		}

		if (!val.unwrap_err()->type()->issubclass(types::stop_iteration())) {
			return Err(val.unwrap_err());
		}

		return Ok(std::monostate{});
	}

	return Err(
		type_error("list indices must be integers or slices, not {}", index->type()->name()));
}

PyResult<size_t> PyList::__len__() const { return Ok(m_elements.size()); }

PyResult<PyObject *> PyList::__add__(const PyObject *other) const
{
	if (!other->type()->issubclass(types::list())) {
		return Err(
			type_error("can only concatenate list (not \"{}\") to list", other->type()->name()));
	}
	const auto &other_list = static_cast<const PyList &>(*other);
	auto result = PyList::create(this->elements());
	if (result.is_err()) { return result; }

	result.unwrap()->elements().insert(result.unwrap()->elements().end(),
		other_list.elements().begin(),
		other_list.elements().end());

	return result;
}

PyResult<PyObject *> PyList::__mul__(size_t count) const
{
	if (count <= 0) { return PyList::create(); }
	std::vector<Value> values;
	values.reserve(count * m_elements.size());
	for (auto _ : std::views::iota(size_t{ 0 }, count)) {
		values.insert(values.end(), m_elements.begin(), m_elements.end());
	}

	return PyList::create(std::move(values));
}

PyResult<PyObject *> PyList::__eq__(const PyObject *other) const
{
	if (!as<PyList>(other)) { return Ok(py_false()); }

	auto *other_list = as<PyList>(other);
	// Value contains PyObject* so we can't just compare vectors with std::vector::operator==
	// otherwise if we compare PyObject* with PyObject* we compare the pointers, rather
	// than PyObject::__eq__(const PyObject*)
	if (m_elements.size() != other_list->elements().size()) { return Ok(py_false()); }
	auto &interpreter = VirtualMachine::the().interpreter();
	const bool result = std::equal(m_elements.begin(),
		m_elements.end(),
		other_list->elements().begin(),
		[&interpreter](const auto &lhs, const auto &rhs) -> bool {
			const auto &result = equals(lhs, rhs, interpreter);
			ASSERT(result.is_ok())
			auto is_true = truthy(result.unwrap(), interpreter);
			ASSERT(is_true.is_ok())
			return is_true.unwrap();
		});
	return Ok(result ? py_true() : py_false());
}

PyResult<PyObject *> PyList::__reversed__() const
{
	return PyListReverseIterator::create(*const_cast<PyList *>(this));
}

PyResult<PyObject *> PyList::sort(PyTuple *args, PyDict *kwargs)
{
	PyObject *key = nullptr;
	bool reverse = false;
	if (args && !args->elements().empty()) {
		return Err(type_error("sort() takes no positional arguments"));
	}
	if (kwargs) {
		if (auto it = kwargs->map().find(String{ "key" }); it != kwargs->map().end()) {
			key = PyObject::from(it->second).unwrap();
		}
		if (auto it = kwargs->map().find(String{ "reverse" }); it != kwargs->map().end()) {
			auto reverse_ = truthy(it->second, VirtualMachine::the().interpreter());
			if (reverse_.is_err()) { return Err(reverse_.unwrap_err()); }
			reverse = reverse_.unwrap();
		}
	}

	PyResult<PyObject *> err = Ok(py_none());
	if (key && key != py_none()) {
		auto cmp_list_ = PyList::create();
		if (cmp_list_.is_err()) { return cmp_list_; }
		auto *cmp_list = cmp_list_.unwrap();

		for (const auto &el : m_elements) {
			auto cmp_value = key->call(PyTuple::create({ el }).unwrap(), nullptr);
			if (cmp_value.is_err()) { return cmp_value; }
			cmp_list->elements().push_back(cmp_value.unwrap());
		}
		std::vector<size_t> indices(cmp_list->elements().size());
		std::iota(indices.begin(), indices.end(), 0);
		// FIXME: should throw exception when comparing, as returning true is
		// probably messing up the C++ Compare requirment
		auto cmp = [&err, cmp_list](size_t lhs_index, size_t rhs_index) -> bool {
			const auto &lhs = cmp_list->elements()[lhs_index];
			const auto &rhs = cmp_list->elements()[rhs_index];
			if (auto cmp = less_than(lhs, rhs, VirtualMachine::the().interpreter()); cmp.is_ok()) {
				auto is_true = truthy(cmp.unwrap(), VirtualMachine::the().interpreter());
				if (is_true.is_err()) {
					err = Err(is_true.unwrap_err());
					return true;
				}
				return is_true.unwrap();
			} else {
				return false;
			}
		};
		if (reverse) {
			std::stable_sort(indices.rbegin(), indices.rend(), cmp);
		} else {
			std::stable_sort(indices.begin(), indices.end(), cmp);
		}

		if (err.is_err()) { return err; }

		for (size_t i = 0; i < indices.size() - 1; ++i) {
			if (indices[i] == i) { continue; }
			size_t o = i + 1;
			for (; o < indices.size(); ++o) {
				if (indices[o] == i) { break; }
			}
			std::iter_swap(m_elements.begin() + i, m_elements.begin() + indices[i]);
			std::iter_swap(indices.begin() + i, indices.begin() + o);
		}
	} else {
		// FIXME: should throw exception when comparing, as returning true is
		// probably messing up the C++ Compare requirment
		auto cmp = [&err](const Value &lhs, const Value &rhs) -> bool {
			if (auto cmp = less_than(lhs, rhs, VirtualMachine::the().interpreter()); cmp.is_ok()) {
				auto is_true = truthy(cmp.unwrap(), VirtualMachine::the().interpreter());
				if (is_true.is_err()) {
					err = Err(is_true.unwrap_err());
					return true;
				}
				return is_true.unwrap();
			} else {
				return false;
			}
		};
		if (reverse) {
			std::stable_sort(m_elements.rbegin(), m_elements.rend(), cmp);
		} else {
			std::stable_sort(m_elements.begin(), m_elements.end(), cmp);
		}

		if (err.is_err()) { return err; }
	}

	return err;
}

void PyList::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto &el : m_elements) {
		if (std::holds_alternative<PyObject *>(el)) {
			if (std::get<PyObject *>(el) != this) visitor.visit(*std::get<PyObject *>(el));
		}
	}
}

PyType *PyList::static_type() const { return types::list(); }

namespace {

	std::once_flag list_flag;

	std::unique_ptr<TypePrototype> register_list()
	{
		return std::move(
			klass<PyList>("list")
				.def("append", &PyList::append)
				.def("extend", &PyList::extend)
				.def(
					"pop",
					+[](PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
						auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
							kwargs,
							"pop",
							std::integral_constant<size_t, 0>{},
							std::integral_constant<size_t, 1>{},
							nullptr);
						if (result.is_err()) return Err(result.unwrap_err());
						return static_cast<PyList *>(self)->pop(std::get<0>(result.unwrap()));
					})
				.def("sort", &PyList::sort)
				.classmethod(
					"__class_getitem__",
					+[](PyType *type, PyTuple *args, PyDict *kwargs) {
						ASSERT(args && args->elements().size() == 1);
						ASSERT(!kwargs || kwargs->map().empty());
						return PyObject::from(args->elements()[0]).and_then([type](PyObject *arg) {
							return PyGenericAlias::create(type, arg);
						});
					})
				.def("__reversed__", &PyList::__reversed__)
				.type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyList::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(list_flag, []() { type = register_list(); });
		return std::move(type);
	};
}


PyListIterator::PyListIterator(const PyList &pylist)
	: PyBaseObject(types::BuiltinTypes::the().list_iterator()), m_pylist(pylist)
{}

std::string PyListIterator::to_string() const
{
	return fmt::format("<list_iterator at {}>", static_cast<const void *>(this));
}

void PyListIterator::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	// the iterator has to keep a reference to the list
	// otherwise GC could clean up a temporary list in a loop
	// TODO: should visit_graph be const and the bit flags mutable?
	visitor.visit(const_cast<PyList &>(m_pylist));
}

PyResult<PyObject *> PyListIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyListIterator::__next__()
{
	if (m_current_index < m_pylist.elements().size())
		return std::visit([](const auto &element) { return PyObject::from(element); },
			m_pylist.elements()[m_current_index++]);
	return Err(stop_iteration());
}

PyType *PyListIterator::static_type() const { return types::list_iterator(); }

namespace {

	std::once_flag list_iterator_flag;

	std::unique_ptr<TypePrototype> register_list_iterator()
	{
		return std::move(klass<PyListIterator>("list_iterator").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyListIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(list_iterator_flag, []() { type = register_list_iterator(); });
		return std::move(type);
	};
}

PyListReverseIterator::PyListReverseIterator(PyType *type) : PyBaseObject(type) {}

PyListReverseIterator::PyListReverseIterator(PyList &pylist, size_t start_index)
	: PyBaseObject(types::BuiltinTypes::the().list_reverseiterator()), m_pylist(pylist),
	  m_current_index(start_index)
{}

PyResult<PyListReverseIterator *> PyListReverseIterator::create(PyList &lst)
{
	auto list_size = lst.elements().size();
	auto *result = VirtualMachine::the().heap().allocate<PyListReverseIterator>(lst, list_size - 1);
	if (!result) { return Err(memory_error(sizeof(PyListReverseIterator))); }
	return Ok(result);
}

void PyListReverseIterator::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_pylist.has_value()) { visitor.visit(m_pylist->get()); }
}

PyResult<PyObject *> PyListReverseIterator::__iter__() const
{
	return Ok(const_cast<PyListReverseIterator *>(this));
}

PyResult<PyObject *> PyListReverseIterator::__next__()
{
	if (m_pylist.has_value()) {
		if (m_current_index < m_pylist->get().elements().size())
			return std::visit([](const auto &element) { return PyObject::from(element); },
				m_pylist->get().elements()[m_current_index--]);
		m_pylist = std::nullopt;
	}
	return Err(stop_iteration());
}

PyType *PyListReverseIterator::static_type() const { return types::list_reverseiterator(); }

namespace {

	std::once_flag list_reverseiterator_flag;

	std::unique_ptr<TypePrototype> register_list_reverseiterator()
	{
		return std::move(klass<PyListReverseIterator>("list_reverseiterator").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyListReverseIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(list_reverseiterator_flag, []() { type = register_list_reverseiterator(); });
		return std::move(type);
	};
}
}// namespace py
