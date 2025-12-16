#include "PyByteArray.hpp"
#include "MemoryError.hpp"
#include "PyBytes.hpp"
#include "StopIteration.hpp"
#include "runtime/IndexError.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PySlice.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/Value.hpp"
#include "runtime/ValueError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "utilities.hpp"
#include "vm/VM.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <numeric>
#include <variant>

namespace py {
namespace {
	static constexpr std::array<std::byte, 0> kEmptyByteArray = {};
}

template<> PyByteArray *as(PyObject *obj)
{
	if (obj->type() == types::bytearray()) { return static_cast<PyByteArray *>(obj); }
	return nullptr;
}

template<> const PyByteArray *as(const PyObject *obj)
{
	if (obj->type() == types::bytearray()) { return static_cast<const PyByteArray *>(obj); }
	return nullptr;
}

PyByteArray::PyByteArray(PyType *type) : PyBaseObject(type) {}

PyByteArray::PyByteArray(const Bytes &value)
	: PyBaseObject(types::BuiltinTypes::the().bytearray()), m_value(value)
{}

PyResult<PyByteArray *> PyByteArray::create(const Bytes &bytes)
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate<PyByteArray>(bytes);
	if (!obj) { return Err(memory_error(sizeof(PyByteArray))); }
	return Ok(obj);
}

PyResult<PyByteArray *> PyByteArray::create() { return PyByteArray::create({}); }

PyResult<PyObject *> PyByteArray::__new__(const PyType *type, PyTuple *, PyDict *)
{
	ASSERT(type == types::bytearray());
	return PyByteArray::create();
}

struct ByteBackInserter
{
	using iterator_category = std::output_iterator_tag;
	using value_type = void;
	using difference_type = std::ptrdiff_t;
	using pointer = void;
	using reference = void;
	using container_type = std::vector<std::byte>;

	container_type &m_bytes;
	BaseException *m_exception{ nullptr };
	ByteBackInserter(std::vector<std::byte> &bytes) : m_bytes(bytes) {}

	BaseException *last_error() const { return m_exception; }

	ByteBackInserter &operator=(PyObject *value)
	{
		if (auto int_obj = as<PyInteger>(value)) {
			if (int_obj->as_i64() >= 0 && int_obj->as_i64() <= 255) {
				m_bytes.push_back(static_cast<std::byte>(int_obj->as_i64()));
			}
		} else {
			m_exception = type_error(
				"'{}' object cannot be interpreted as an integer", value->type()->name());
		}
		return *this;
	}
};

static_assert(detail::has_output_iterator_error<ByteBackInserter>);

PyResult<int32_t> PyByteArray::__init__(PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty());

	if (!args || args->elements().empty()) {
		return Ok(0);
	} else if (args->elements().size() == 1) {
		auto arg0 = PyObject::from(args->elements()[0]);
		if (arg0.is_err()) { return Err(arg0.unwrap_err()); }
		if (auto count = as<PyInteger>(arg0.unwrap())) {
			m_value.b.resize(count->as_size_t());
		} else if (auto bytes = as<PyBytes>(arg0.unwrap())) {
			// FIXME: should this take the iterable path?
			m_value.b.insert(m_value.b.end(), bytes->value().b.begin(), bytes->value().b.end());
		} else if (arg0.unwrap()->iter().is_ok()) {
			if (auto result = from_iterable(arg0.unwrap(), ByteBackInserter(m_value.b));
				result.is_err()) {
				return Err(result.unwrap_err());
			}
		} else {
			TODO();
		}
	} else {
		TODO();
	}

	return Ok(0);
}

std::string PyByteArray::to_string() const
{
	std::ostringstream os;
	os << "bytearray(" << m_value.to_string() << ")";
	return os.str();
}

PyResult<size_t> PyByteArray::__len__() const { return Ok(m_value.b.size()); }

PyResult<PyObject *> PyByteArray::__iter__() const
{
	return PyByteArrayIterator::create(const_cast<PyByteArray *>(this));
}

PyResult<PyObject *> PyByteArray::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyByteArray::__getitem__(int64_t index)
{
	if (index < 0) { index += m_value.b.size(); }
	if (index < 0 || static_cast<size_t>(index) >= m_value.b.size()) {
		return Err(index_error("bytearray index out of range"));
	}
	return PyInteger::create(static_cast<int64_t>(m_value.b[index]));
}

PyResult<std::monostate> PyByteArray::__setitem__(int64_t index, PyObject *value)
{
	if (index < 0) { index += m_value.b.size(); }
	if (index < 0 || static_cast<size_t>(index) >= m_value.b.size()) {
		return Err(index_error("bytearray index out of range"));
	}

	if (!value->type()->issubclass(types::integer())) {
		return Err(
			type_error("'{}' object cannot be interpreted as an integer", value->type()->name()));
	}

	auto new_value = static_cast<const PyInteger &>(*value).as_big_int();

	if (new_value < 0 || new_value > 255) {
		return Err(value_error("byte must be in range(0, 256)"));
	}

	m_value.b[index] = static_cast<std::byte>(new_value.get_ui());

	return Ok(std::monostate{});
}

PyResult<PyObject *> PyByteArray::__getitem__(PyObject *index)
{
	if (index->type()->issubclass(types::integer())) {
		const auto i = static_cast<const PyInteger &>(*index).as_i64();
		return __getitem__(i);
	} else if (index->type()->issubclass(types::slice())) {
		auto slice = static_cast<PySlice *>(index);
		auto indices_ = slice->unpack();
		if (indices_.is_err()) return Err(indices_.unwrap_err());
		const auto [start_, end_, step] = indices_.unwrap();

		const auto [start, end, slice_length] =
			PySlice::adjust_indices(start_, end_, step, m_value.b.size());

		if (slice_length == 0) { return PyByteArray::create(); }
		if (start == 0 && end == static_cast<int64_t>(m_value.b.size()) && step == 1) {
			return PyByteArray::create(m_value);
		}

		Bytes bytes;
		bytes.b.reserve(slice_length);
		for (int64_t idx = start, i = 0; i < slice_length; idx += step, ++i) {
			bytes.b.push_back(m_value.b[idx]);
		}

		return PyByteArray::create(bytes);
	} else {
		return Err(type_error(
			"bytearray indices must be integers or slices, not {}", index->type()->name()));
	}
}


PyResult<std::monostate> PyByteArray::__setitem__(PyObject *index, PyObject *value)
{
	if (index->type()->issubclass(types::integer())) {
		const auto i = static_cast<const PyInteger &>(*index).as_i64();
		return __setitem__(i, value);
	} else if (index->type()->issubclass(types::slice())) {
		const auto &slice = static_cast<const PySlice &>(*index);
		auto indices_ = slice.unpack();
		const auto [start_, end_, step] = indices_.unwrap();

		const auto [start, stop, slice_length] =
			PySlice::adjust_indices(start_, end_, step, m_value.b.size());

		if (step == 0) { return Err(value_error("slice step cannot be zero")); }
		if (slice_length == 0) { return Ok(std::monostate{}); }
		if (start > stop && step > 0) { return Ok(std::monostate{}); }
		if (start > static_cast<int64_t>(m_value.b.size()) || start < 0) {
			return Ok(std::monostate{});
		}

		if (step != 1) { TODO(); }

		if (value->type()->issubclass(types::bytes())
			|| value->type()->issubclass(types::bytearray())) {
			Bytes bytes;
			if (value->type()->issubclass(types::bytes())) {
				bytes = static_cast<const PyBytes &>(*value).value();
			} else {
				bytes = static_cast<const PyByteArray &>(*value).value();
			}

			// naive implementation, we just remove values, and then insert new ones
			auto it = m_value.b.erase(m_value.b.begin() + start, m_value.b.begin() + stop);
			m_value.b.insert(it, bytes.b.begin(), bytes.b.end());

			return Ok(std::monostate{});
		}
		auto value_iter = value->iter();
		if (value_iter.is_err()) {
			return Err(type_error(
				"can assign only bytes, buffers, or iterables of ints in range(0, 256)"));
		}

		TODO();
	}

	return Err(
		type_error("bytearray indices must be integers or slices, not {}", index->type()->name()));
}

PyResult<std::monostate> PyByteArray::__getbuffer__(PyBuffer &view, int)
{
	view.obj = this;
	if (m_value.b.empty()) {
		view.buf = std::make_unique<NonOwningStorage<std::byte>>(
			const_cast<std::byte *>(kEmptyByteArray.data()));
	} else {
		view.buf = std::make_unique<NonOwningStorage<std::byte>>(m_value.b.data());
	}
	view.len = m_value.b.size();
	view.readonly = false;
	view.itemsize = 1;
	view.format = "B";
	view.ndim = 1;
	return Ok(std::monostate{});
}

PyResult<std::monostate> PyByteArray::__releasebuffer__(PyBuffer &) { return Ok(std::monostate{}); }

PyResult<PyObject *> PyByteArray::__add__(const PyObject *other) const
{
	auto new_bytes = m_value;
	if (auto bytes = as<PyBytes>(other)) {
		new_bytes.b.insert(new_bytes.b.end(), bytes->value().b.begin(), bytes->value().b.end());
	} else if (auto bytearray = as<PyByteArray>(other)) {
		new_bytes.b.insert(
			new_bytes.b.end(), bytearray->value().b.begin(), bytearray->value().b.end());
	} else {
		return Err(type_error("can't concat {} to bytes", other->type()->name()));
	}

	return PyByteArray::create(new_bytes);
}

PyResult<PyObject *> PyByteArray::find(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() <= 3 && args->size() > 0);
	ASSERT(!kwargs);

	auto pattern_ = PyObject::from(args->elements()[0]);
	if (pattern_.is_err()) return pattern_;
	if (!pattern_.unwrap()->type()->issubclass(types::integer())) { TODO(); }
	const auto &pattern_int = static_cast<const PyInteger &>(*pattern_.unwrap());
	if (pattern_int.as_big_int() < 0 && pattern_int.as_big_int() > 255) { TODO(); }
	PyInteger *start = nullptr;
	PyInteger *end = nullptr;
	int64_t result = -1;

	if (args->size() >= 2) {
		auto start_ = PyObject::from(args->elements()[1]);
		if (start_.is_err()) return start_;
		start = as<PyInteger>(start_.unwrap());
		// TODO: raise exception when start in not a number
		ASSERT(start);
	}
	if (args->size() == 3) {
		auto end_ = PyObject::from(args->elements()[2]);
		if (end_.is_err()) return end_;
		end = as<PyInteger>(end_.unwrap());
		// TODO: raise exception when end in not a number
		ASSERT(end);
	}

	auto get_position_from_slice = [this](int64_t pos) -> PyResult<size_t> {
		if (pos < 0) {
			pos += m_value.b.size();
			// TODO: handle case where the negative start index is less than size of string
			if (pos < 0) { return Err(index_error("bytearray index out of range")); }
		}
		return Ok(pos);
	};

	auto find = [value = static_cast<std::byte>(pattern_int.as_size_t())](
					std::span<const std::byte> bytes, size_t start) -> int64_t {
		auto it = std::ranges::find(bytes, value);
		if (it == bytes.end()) { return -1; }
		return start + std::distance(bytes.begin(), it);
	};

	if (!start && !end) {
		result = find(m_value.b, 0);
	} else if (!end) {
		auto start_ =
			std::visit(overloaded{
						   [get_position_from_slice](const auto &val) -> PyResult<size_t> {
							   return get_position_from_slice(static_cast<int64_t>(val));
						   },
						   [get_position_from_slice](const mpz_class &val) -> PyResult<size_t> {
							   ASSERT(val.fits_slong_p());
							   return get_position_from_slice(val.get_si());
						   },
					   },
				start->value().value);
		if (start_.is_err()) { return Err(start_.unwrap_err()); }
		result = find(
			std::span{ m_value.b.begin() + start_.unwrap(), m_value.b.end() }, start_.unwrap());
	} else {
		auto start_ =
			std::visit(overloaded{
						   [get_position_from_slice](const auto &val) -> PyResult<size_t> {
							   return get_position_from_slice(static_cast<int64_t>(val));
						   },
						   [get_position_from_slice](const mpz_class &val) -> PyResult<size_t> {
							   ASSERT(val.fits_slong_p());
							   return get_position_from_slice(val.get_si());
						   },
					   },
				start->value().value);
		if (start_.is_err()) { return Err(start_.unwrap_err()); }
		auto end_ =
			std::visit(overloaded{
						   [get_position_from_slice](const auto &val) -> PyResult<size_t> {
							   return get_position_from_slice(static_cast<int64_t>(val));
						   },
						   [get_position_from_slice](const mpz_class &val) -> PyResult<size_t> {
							   ASSERT(val.fits_slong_p());
							   return get_position_from_slice(val.get_si());
						   },
					   },
				end->value().value);
		if (end_.is_err()) { return Err(end_.unwrap_err()); }
		result = find(
			std::span{ m_value.b.begin() + start_.unwrap(), m_value.b.begin() + end_.unwrap() },
			start_.unwrap());
	}

	return PyInteger::create(result);
}

PyResult<PyObject *> PyByteArray::maketrans(PyObject *from, PyObject *to)
{
	Bytes from_bytes;
	if (from->type()->issubclass(types::bytes())) {
		from_bytes = static_cast<const PyBytes &>(*from).value();
	} else if (from->type()->issubclass(types::bytearray())) {
		from_bytes = static_cast<const PyByteArray &>(*from).value();
	} else {
		return Err(type_error("a bytes-like object is required, not '{}'", from->type()->name()));
	}

	Bytes to_bytes;
	if (to->type()->issubclass(types::bytes())) {
		to_bytes = static_cast<const PyBytes &>(*to).value();
	} else if (to->type()->issubclass(types::bytearray())) {
		to_bytes = static_cast<const PyByteArray &>(*to).value();
	} else {
		return Err(type_error("a bytes-like object is required, not '{}'", from->type()->name()));
	}

	if (from_bytes.b.size() != to_bytes.b.size()) {
		return Err(value_error("maketrans arguments must have same length"));
	}

	Bytes result;
	result.b.reserve(256);
	for (size_t i = 0; i < 256; ++i) { result.b.push_back(static_cast<std::byte>(i)); }
	for (size_t i = 0; i < 256; ++i) {
		const auto from_byte = from_bytes.b[i];
		const auto to_byte = to_bytes.b[i];
		result.b[static_cast<size_t>(from_byte)] = to_byte;
	}

	return PyByteArray::create(result);
}

PyResult<PyObject *> PyByteArray::translate(PyTuple *args, PyDict *kwargs) const
{
	Bytes table;
	Bytes to_delete;
	if (!args || args->size() == 0) {
		return Err(type_error(
			"translate() takes at least 1 positional argument ({} given)", args->size()));
	}
	if (auto argcount = (args->size() + (kwargs ? kwargs->size() : 0)); argcount > 2) {
		return Err(type_error("translate() takes at most 2 arguments ({} given)", argcount));
	}
	auto el0 = PyObject::from(args->elements()[0]).unwrap();
	if (el0->type()->issubclass(types::bytes())) {
		table = static_cast<const PyBytes &>(*el0).value();
	} else if (el0->type()->issubclass(types::bytearray())) {
		table = static_cast<const PyByteArray &>(*el0).value();
	} else if (el0 != py_none()) {
		return Err(type_error("a bytes-like object is required, not '{}'", el0->type()->name()));
	}

	if (args->size() == 2) {
		auto *el1 = PyObject::from(args->elements()[1]).unwrap();
		if (el1->type()->issubclass(types::bytes())) {
			to_delete = static_cast<const PyBytes &>(*el1).value();
		} else if (el1->type()->issubclass(types::bytearray())) {
			to_delete = static_cast<const PyByteArray &>(*el1).value();
		} else {
			return Err(
				type_error("a bytes-like object is required, not '{}'", el1->type()->name()));
		}
	} else if (kwargs) {
		if (auto it = kwargs->map().find(String{ "delete" }); it != kwargs->map().end()) {
			auto *el1 = PyObject::from(it->second).unwrap();
			if (el1->type()->issubclass(types::bytes())) {
				to_delete = static_cast<const PyBytes &>(*el1).value();
			} else if (el1->type()->issubclass(types::bytearray())) {
				to_delete = static_cast<const PyByteArray &>(*el1).value();
			} else {
				return Err(
					type_error("a bytes-like object is required, not '{}'", el1->type()->name()));
			}
		}
	}

	if (!table.b.empty() && table.b.size() != 256) {
		return Err(value_error("translation table must be 256 characters long"));
	}

	Bytes result;
	std::ranges::remove_copy_if(
		m_value.b, std::back_inserter(result.b), [&to_delete](const auto &el) -> bool {
			return std::ranges::find(to_delete.b, el) != to_delete.b.end();
		});

	if (!table.b.empty()) {
		std::ranges::transform(result.b, result.b.begin(), [&table](const auto el) {
			return table.b[static_cast<size_t>(el)];
		});
	}

	return PyByteArray::create(result);
}

PyResult<PyObject *> PyByteArray::__eq__(const PyObject *other) const
{
	if (other->type()->issubclass(types::bytearray())) {
		return Ok(m_value.b == static_cast<const PyByteArray &>(*other).value().b ? py_true()
																				  : py_false());
	} else if (other->type()->issubclass(types::bytes())) {
		return Ok(
			m_value.b == static_cast<const PyBytes &>(*other).value().b ? py_true() : py_false());
	}
	return Ok(py_false());
}


namespace {

	std::once_flag bytearray_flag;

	std::unique_ptr<TypePrototype> register_bytearray()
	{
		return std::move(klass<PyByteArray>("bytearray")
				.def("find", &PyByteArray::find)
				.def("translate", &PyByteArray::translate)
				.staticmethod("maketrans", &PyByteArray::maketrans)
				.type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyByteArray::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(bytearray_flag, []() { type = register_bytearray(); });
		return std::move(type);
	};
}

PyType *PyByteArray::static_type() const { return types::bytearray(); }

PyByteArrayIterator::PyByteArrayIterator(PyType *type) : PyBaseObject(type) {}

PyByteArrayIterator::PyByteArrayIterator(PyByteArray *bytes, size_t index)
	: PyBaseObject(types::BuiltinTypes::the().bytearray_iterator()), m_bytes(bytes), m_index(index)
{}

PyResult<PyByteArrayIterator *> PyByteArrayIterator::create(PyByteArray *bytes_array)
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate<PyByteArrayIterator>(bytes_array, 0);
	if (!obj) { return Err(memory_error(sizeof(PyByteArrayIterator))); }
	return Ok(obj);
}

std::string PyByteArrayIterator::to_string() const
{
	return fmt::format("<bytearray_iterator object at {}>", static_cast<const void *>(this));
}

PyResult<PyObject *> PyByteArrayIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyByteArrayIterator::__next__()
{
	if (!m_bytes || m_index >= m_bytes->value().b.size()) { return Err(stop_iteration()); }
	const auto value = m_bytes->value().b[m_index++];
	return PyInteger::create(static_cast<int64_t>(value));
}

namespace {

	std::once_flag bytearray_iterator_flag;

	std::unique_ptr<TypePrototype> register_bytearray_iterator()
	{
		return std::move(klass<PyByteArrayIterator>("bytearray_iterator").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyByteArrayIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(bytearray_iterator_flag, []() { type = register_bytearray_iterator(); });
		return std::move(type);
	};
}

PyType *PyByteArrayIterator::static_type() const { return types::bytearray_iterator(); }

void PyByteArrayIterator::visit_graph(Visitor &visitor)
{
	if (m_bytes) { visitor.visit(*m_bytes); }
}

}// namespace py
