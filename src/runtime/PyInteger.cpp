#include "PyInteger.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

PyInteger::PyInteger(int64_t value)
	: PyNumber(Number{ value }, PyObjectType::PY_INTEGER, BuiltinTypes::the().integer())
{}

PyInteger *PyInteger::create(int64_t value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyInteger>(value);
}

PyType *PyInteger::type_() const { return integer(); }

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