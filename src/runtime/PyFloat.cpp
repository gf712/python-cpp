#include "PyFloat.hpp"
#include "MemoryError.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/Value.hpp"
#include "runtime/forward.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

#include <cmath>

namespace py {

template<> PyFloat *as(PyObject *obj)
{
	if (obj->type() == types::float_()) { return static_cast<PyFloat *>(obj); }
	return nullptr;
}

template<> const PyFloat *as(const PyObject *obj)
{
	if (obj->type() == types::float_()) { return static_cast<const PyFloat *>(obj); }
	return nullptr;
}

PyFloat::PyFloat(PyType *type) : PyNumber(type) {}

PyFloat::PyFloat(double value) : PyNumber(Number{ value }, types::BuiltinTypes::the().float_()) {}

PyResult<PyObject *> PyFloat::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	// TODO: support inheriting from float
	ASSERT(type == types::float_());

	ASSERT(!kwargs || kwargs->map().empty())
	PyObject *value = nullptr;
	if (args->elements().size() > 0) {
		if (auto obj = PyObject::from(args->elements()[0]); obj.is_ok()) {
			value = obj.unwrap();
		} else {
			return obj;
		}
	}

	if (!value) {
		return PyFloat::create(0.0);
	} else if (auto *int_value = as<PyInteger>(value)) {
		return PyFloat::create(static_cast<double>(int_value->as_size_t()));
	} else if (auto *float_value = as<PyFloat>(value)) {
		return PyFloat::create(float_value->as_f64());
	}
	TODO();
	return Err(nullptr);
}

PyResult<PyFloat *> PyFloat::create(double value)
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate<PyFloat>(value);
	if (!obj) { return Err(memory_error(sizeof(PyFloat))); }
	return Ok(obj);
}

PyResult<PyObject *> PyFloat::__round__(PyObject *ndigits_obj) const
{
	if (!ndigits_obj || ndigits_obj == py_none()) {
		return PyInteger::create(BigIntType{ as_f64() });
	}

	if (!ndigits_obj->type()->issubclass(types::integer())) {
		return Err(type_error(
			"'{}' object cannot be interpreted as an integer", ndigits_obj->type()->name()));
	}

	auto ndigits = static_cast<const PyInteger &>(*ndigits_obj).as_big_int();

	const auto multiplier = std::pow(10., ndigits.get_d());
	const auto value = std::floor(as_f64() * multiplier) / multiplier;
	return PyFloat::create(value);
}


PyType *PyFloat::static_type() const { return types::float_(); }

double PyFloat::as_f64() const
{
	ASSERT(std::holds_alternative<double>(m_value.value));
	return std::get<double>(m_value.value);
}

namespace {

	std::once_flag float_flag;

	std::unique_ptr<TypePrototype> register_float()
	{
		return std::move(klass<PyFloat>("float").def("__round__", &PyFloat::__round__).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyFloat::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(float_flag, []() { type = register_float(); });
		return std::move(type);
	};
}

}// namespace py
