#include "PyBytes.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"


PyBytes::PyBytes(const Bytes &number)
	: PyBaseObject(PyObjectType::PY_BYTES, BuiltinTypes::the().bytes()), m_value(number)
{}

PyBytes *PyBytes::create(const Bytes &value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyBytes>(value);
}

std::string PyBytes::to_string() const
{
	std::ostringstream os;
	os << m_value;
	return fmt::format("PyBytes {}", os.str());
}

PyObject *PyBytes::__add__(const PyObject *) const { TODO() }

PyType *PyBytes::type_() const
{
	return bytes();
}

namespace {

std::once_flag bytes_flag;

std::unique_ptr<TypePrototype> register_bytes() { return std::move(klass<PyBytes>("bytes").type); }
}// namespace

std::unique_ptr<TypePrototype> PyBytes::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(bytes_flag, []() { type = ::register_bytes(); });
	return std::move(type);
}
