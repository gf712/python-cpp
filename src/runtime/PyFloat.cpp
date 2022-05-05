#include "PyFloat.hpp"
#include "MemoryError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

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

PyFloat::PyFloat(double value) : PyNumber(Number{ value }, BuiltinTypes::the().float_()) {}

PyResult<PyFloat *> PyFloat::create(double value)
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate<PyFloat>(value);
	if (!obj) { return Err(memory_error(sizeof(PyFloat))); }
	return Ok(obj);
}

PyType *PyFloat::type() const { return float_(); }

double PyFloat::as_f64() const
{
	ASSERT(std::holds_alternative<double>(m_value.value));
	return std::get<double>(m_value.value);
}

namespace {

	std::once_flag float_flag;

	std::unique_ptr<TypePrototype> register_float()
	{
		return std::move(klass<PyFloat>("float").type);
	}
}// namespace

std::unique_ptr<TypePrototype> PyFloat::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(float_flag, []() { type = register_float(); });
	return std::move(type);
}

}// namespace py