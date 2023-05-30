#include "RuntimeError.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

template<> RuntimeError *as(PyObject *obj)
{
	ASSERT(types::runtime_error());
	if (obj->type() == types::runtime_error()) { return static_cast<RuntimeError *>(obj); }
	return nullptr;
}

template<> const RuntimeError *as(const PyObject *obj)
{
	ASSERT(types::runtime_error())
	if (obj->type() == types::runtime_error()) { return static_cast<const RuntimeError *>(obj); }
	return nullptr;
}

RuntimeError::RuntimeError(PyType *type) : Exception(type) {}

RuntimeError::RuntimeError(PyTuple *args)
	: Exception(types::BuiltinTypes::the().runtime_error(), args)
{}

PyResult<RuntimeError *> RuntimeError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate<RuntimeError>(args);
	if (!obj) { return Err(memory_error(sizeof(RuntimeError))); }
	return Ok(obj);
}

PyType *RuntimeError::static_type() const
{
	ASSERT(types::runtime_error());
	return types::runtime_error();
}

namespace {

	std::once_flag runtime_error_flag;

	std::unique_ptr<TypePrototype> register_runtime_error()
	{
		return std::move(klass<RuntimeError>("RuntimeError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> RuntimeError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(runtime_error_flag, []() { type = register_runtime_error(); });
		return std::move(type);
	};
}

}// namespace py
