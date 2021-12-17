#include "PyInteger.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

PyInteger::PyInteger(int64_t value) : PyNumber(Number{ value }, BuiltinTypes::the().integer()) {}

PyInteger *PyInteger::create(int64_t value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyInteger>(value);
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
