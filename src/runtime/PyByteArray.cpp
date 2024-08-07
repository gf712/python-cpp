#include "PyByteArray.hpp"
#include "MemoryError.hpp"
#include "PyBytes.hpp"
#include "StopIteration.hpp"
#include "runtime/IndexError.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/ValueError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "utilities.hpp"
#include "vm/VM.hpp"

namespace py {
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
	os << "bytearray(b'";
	os << m_value.to_string();
	os << "')";
	return os.str();
}

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


namespace {

	std::once_flag bytearray_flag;

	std::unique_ptr<TypePrototype> register_bytearray()
	{
		return std::move(klass<PyByteArray>("bytearray").type);
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
