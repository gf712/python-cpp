#include "PyInteger.hpp"
#include "MemoryError.hpp"
#include "PyBytes.hpp"
#include "PyFloat.hpp"
#include "TypeError.hpp"
#include "ValueError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyInteger *as(PyObject *obj)
{
	if (obj->type() == integer()) { return static_cast<PyInteger *>(obj); }
	return nullptr;
}

template<> const PyInteger *as(const PyObject *obj)
{
	if (obj->type() == integer()) { return static_cast<const PyInteger *>(obj); }
	return nullptr;
}


PyInteger::PyInteger(int64_t value) : Interface(Number{ value }, BuiltinTypes::the().integer()) {}

PyInteger::PyInteger(TypePrototype &type, int64_t value) : Interface(Number{ value }, type) {}

PyResult<PyInteger *> PyInteger::create(int64_t value)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<PyInteger>(value);
	if (!result) { return Err(memory_error(sizeof(PyInteger))); }
	return Ok(result);
}

PyResult<PyObject *> PyInteger::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == integer());

	ASSERT(!kwargs || kwargs->map().empty())
	PyObject *value = nullptr;
	PyObject *base = nullptr;
	if (args->elements().size() > 0) {
		if (auto obj = PyObject::from(args->elements()[0]); obj.is_ok()) {
			value = obj.unwrap();
		} else {
			return obj;
		}
	}

	if (args->elements().size() > 1) {
		(void)base;
		TODO();
	}

	if (auto *int_value = as<PyInteger>(value)) {
		return PyInteger::create(int_value->as_size_t());
	} else if (auto *float_value = as<PyFloat>(value)) {
		return PyInteger::create(static_cast<int64_t>(float_value->as_f64()));
	} else if (auto *str_value = as<PyString>(value)) {
		size_t pos{ 0 };
		auto str = str_value->value();
		std::erase_if(str, [](const auto &c) { return std::isspace(c); });
		double result = std::stod(str, &pos);
		if (pos != str.size()) { return Err(type_error("invalid literal for int(): '{}'", str)); }
		return PyInteger::create(static_cast<int64_t>(result));
	}
	TODO();
	return Err(nullptr);
}

PyResult<int64_t> PyInteger::__hash__() const
{
	const auto value = as_i64();
	if (value == -1) return Ok(-2);
	return Ok(value);
}

PyResult<PyObject *> PyInteger::__and__(PyObject *obj)
{
	if (obj->type() != integer()) {
		return Err(type_error("unsupported operand type(s) for &: 'int' and '{}'", obj->type()->name()));
	}

	return PyInteger::create((as_i64() & as<PyInteger>(obj)->as_i64() ));
}

PyResult<PyObject *> PyInteger::__or__(PyObject *obj)
{
	if (obj->type() != integer()) {
		return Err(type_error("unsupported operand type(s) for |: 'int' and '{}'", obj->type()->name()));
	}

	return PyInteger::create((as_i64() | as<PyInteger>(obj)->as_i64() ));
}

PyResult<PyObject *> PyInteger::to_bytes(PyTuple *args, PyDict *kwargs) const
{
	// FIXME: fix signature to to_bytes(length, byteorder, *, signed=False)
	ASSERT(!kwargs || kwargs->map().empty());

	if (args->size() != 2) { return Err(type_error("to_bytes expected two arguments")); }

	auto length_ = PyObject::from(args->elements()[0]);
	if (length_.is_err()) return length_;
	auto byteorder_ = PyObject::from(args->elements()[1]);
	if (byteorder_.is_err()) return byteorder_;

	if (!as<PyInteger>(length_.unwrap())) {
		return Err(type_error(
			"'{}' object cannot be interpreted as an integer", length_.unwrap()->type()->name()));
	}
	if (!as<PyString>(byteorder_.unwrap())) {
		return Err(type_error("to_bytes() argument 'byteorder' must be str, not {}",
			byteorder_.unwrap()->type()->name()));
	}

	if (as<PyInteger>(length_.unwrap())->as_i64() < 0) {
		return Err(type_error("length argument must be non-negative"));
	}
	const auto length = as<PyInteger>(length_.unwrap())->as_size_t();
	const auto byteorder = as<PyString>(byteorder_.unwrap())->value();
	if (byteorder != "little" && byteorder != "big") {
		return Err(value_error("byteorder must be either 'little' or 'big'"));
	}

	std::vector<std::byte> bytes;
	bytes.resize(length, std::byte{ '\0' });
	const std::byte *bytes_ = bit_cast<std::byte *>(&std::get<int64_t>(m_value.value));
	const auto copy_len = std::min(sizeof(int64_t), length);
	if constexpr (std::endian::native == std::endian::big) {
		if (byteorder == "big") {
			std::copy_n(bytes_, copy_len, bytes.begin());
		} else {
			auto it = bytes.rbegin();
			for (size_t i = 0; i < copy_len; i++) { *(it++) = bytes_[i]; }
		}
	} else if constexpr (std::endian::native == std::endian::little) {
		if (byteorder == "little") {
			std::copy_n(bytes_, copy_len, bytes.begin());
		} else {
			auto it = bytes.rbegin();
			for (size_t i = 0; i < copy_len; i++) { *(it++) = bytes_[i]; }
		}
	}

	return PyBytes::create(Bytes{ bytes });
}

PyResult<PyObject *> PyInteger::from_bytes(PyType *type, PyTuple *args, PyDict *kwargs)
{
	// FIXME: fix signature to from_bytes(bytes, byteorder, *, signed=False)
	ASSERT(!kwargs || kwargs->map().empty());

	if (args->size() != 2) { return Err(type_error("from_bytes expected two arguments")); }

	ASSERT(type == integer());

	auto bytes_ = PyObject::from(args->elements()[0]);
	if (bytes_.is_err()) return bytes_;
	auto byteorder_ = PyObject::from(args->elements()[1]);
	if (byteorder_.is_err()) return byteorder_;

	if (!as<PyBytes>(bytes_.unwrap())) {
		return Err(type_error(
			"'{}' object cannot be interpreted as a bytes array", bytes_.unwrap()->type()->name()));
	}
	if (!as<PyString>(byteorder_.unwrap())) {
		return Err(type_error("to_bytes() argument 'byteorder' must be str, not {}",
			byteorder_.unwrap()->type()->name()));
	}

	const auto bytes = as<PyBytes>(bytes_.unwrap());
	const auto byteorder = as<PyString>(byteorder_.unwrap())->value();
	if (byteorder != "little" && byteorder != "big") {
		return Err(value_error("byteorder must be either 'little' or 'big'"));
	}

	if (bytes->value().b.size() > 8) { TODO(); }

	uint64_t value{ 0 };
	if (byteorder != "big") {
		for (size_t i = 0; i < bytes->value().b.size(); ++i) {
			value |= static_cast<uint64_t>(bytes->value().b[i]) << i * 8;
		}
	} else {
		for (size_t i = 0; i < bytes->value().b.size(); ++i) {
			value |= static_cast<uint64_t>(bytes->value().b[i])
					 << ((sizeof(uint64_t) * 8) - ((i - 1) * 8));
		}
	}

	ASSERT(value < static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))

	auto result = PyInteger::create(static_cast<int64_t>(value));
	if (result.is_err()) return result;
	if (type != integer()) {
		return type->__call__(PyTuple::create(result.unwrap()).unwrap(), PyDict::create().unwrap());
	}
	return result;
}

int64_t PyInteger::as_i64() const
{
	ASSERT(std::holds_alternative<int64_t>(m_value.value))
	return std::get<int64_t>(m_value.value);
}

size_t PyInteger::as_size_t() const
{
	ASSERT(std::holds_alternative<int64_t>(m_value.value))
	return static_cast<size_t>(std::get<int64_t>(m_value.value));
}

PyType *PyInteger::type() const { return integer(); }

namespace {

	std::once_flag int_flag;

	std::unique_ptr<TypePrototype> register_int()
	{
		return std::move(klass<PyInteger>("int")
							 .def("to_bytes", &PyInteger::to_bytes)
							 .classmethod("from_bytes", &PyInteger::from_bytes)
							 .type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyInteger::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(int_flag, []() { type = register_int(); });
		return std::move(type);
	};
}
}// namespace py