#include "PyFloat.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"


PyFloat::PyFloat(double value) : PyNumber(Number{ value }, BuiltinTypes::the().float_()) {}

PyFloat *PyFloat::create(double value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyFloat>(value);
}

PyType *PyFloat::type() const { return float_(); }

namespace {

std::once_flag float_flag;

std::unique_ptr<TypePrototype> register_float() { return std::move(klass<PyFloat>("float").type); }
}// namespace

std::unique_ptr<TypePrototype> PyFloat::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(float_flag, []() { type = ::register_float(); });
	return std::move(type);
}

template<> PyFloat *as(PyObject *obj)
{
	if (obj->type() == float_()) { return static_cast<PyFloat *>(obj); }
	return nullptr;
}

template<> const PyFloat *as(const PyObject *obj)
{
	if (obj->type() == float_()) { return static_cast<const PyFloat *>(obj); }
	return nullptr;
}
