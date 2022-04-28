#include "PyBytes.hpp"
#include "MemoryError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyBytes *as(PyObject *obj)
{
	if (obj->type() == bytes()) { return static_cast<PyBytes *>(obj); }
	return nullptr;
}

template<> const PyBytes *as(const PyObject *obj)
{
	if (obj->type() == bytes()) { return static_cast<const PyBytes *>(obj); }
	return nullptr;
}

PyBytes::PyBytes(const Bytes &number) : PyBaseObject(BuiltinTypes::the().bytes()), m_value(number)
{}

PyResult PyBytes::create(const Bytes &value)
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate<PyBytes>(value);
	if (!obj) { return PyResult::Err(memory_error(sizeof(PyBytes))); }
	return PyResult::Ok(obj);
}

std::string PyBytes::to_string() const
{
	std::ostringstream os;
	os << m_value;
	return fmt::format("PyBytes {}", os.str());
}

PyResult PyBytes::__add__(const PyObject *) const { TODO(); }

PyType *PyBytes::type() const { return bytes(); }

namespace {

	std::once_flag bytes_flag;

	std::unique_ptr<TypePrototype> register_bytes()
	{
		return std::move(klass<PyBytes>("bytes").type);
	}
}// namespace

std::unique_ptr<TypePrototype> PyBytes::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(bytes_flag, []() { type = register_bytes(); });
	return std::move(type);
}

}// namespace py