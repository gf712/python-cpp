#include "PyInteger.hpp"
#include "MemoryError.hpp"
#include "PyFloat.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

template<> PyInteger *py::as(PyObject *obj)
{
	if (obj->type() == integer()) { return static_cast<PyInteger *>(obj); }
	return nullptr;
}

template<> const PyInteger *py::as(const PyObject *obj)
{
	if (obj->type() == integer()) { return static_cast<const PyInteger *>(obj); }
	return nullptr;
}


PyInteger::PyInteger(int64_t value) : PyNumber(Number{ value }, BuiltinTypes::the().integer()) {}

PyResult PyInteger::create(int64_t value)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<PyInteger>(value);
	if (!result) { return PyResult::Err(memory_error(sizeof(PyInteger))); }
	return PyResult::Ok(result);
}

PyResult PyInteger::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == integer());

	ASSERT(!kwargs || kwargs->map().empty())
	PyObject *value = nullptr;
	PyObject *base = nullptr;
	if (args->elements().size() > 0) {
		if (auto obj = PyObject::from(args->elements()[0]); obj.is_ok()) {
			value = obj.unwrap_as<PyObject>();
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
		if (pos != str.size()) {
			return PyResult::Err(type_error("invalid literal for int(): '{}'", str));
		}
		return PyInteger::create(static_cast<int64_t>(result));
	}
	TODO();
	return PyResult::Err(nullptr);
}

size_t PyInteger::as_i64() const
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

std::unique_ptr<TypePrototype> register_int() { return std::move(klass<PyInteger>("int").type); }
}// namespace

std::unique_ptr<TypePrototype> PyInteger::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(int_flag, []() { type = ::register_int(); });
	return std::move(type);
}
