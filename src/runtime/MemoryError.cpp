module;
#include "core.hpp"

module py.runtime;
import py.types;


namespace py {

PyResult<MemoryError *> MemoryError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto result = heap.allocate<MemoryError>(args);
	if (!result) {
		// TODO: if this exception fails to allocated we need to find a solution to signal it.
		//       could force a GC run and try again?
		TODO();
	}
	return Ok(result);
}

BaseException *memory_error(std::size_t failed_allocation_size)
{
	// if the allocation of the exception parameters fail, we bail (for now at least)
	auto msg = PyString::create(
		std::format("memory allocation failed, allocating {} bytes", failed_allocation_size));
	if (msg.is_err()) { TODO(); }
	auto args_tuple = PyTuple::create(msg.unwrap());
	if (args_tuple.is_err()) { TODO(); }
	return MemoryError::create(args_tuple.unwrap()).unwrap();
}

template<> MemoryError *as(PyObject *obj)
{
	ASSERT(types::memory_error());
	if (obj->type() == types::memory_error()) { return static_cast<MemoryError *>(obj); }
	return nullptr;
}

template<> const MemoryError *as(const PyObject *obj)
{
	ASSERT(types::memory_error());
	if (obj->type() == types::memory_error()) { return static_cast<const MemoryError *>(obj); }
	return nullptr;
}

MemoryError::MemoryError(PyType *type) : Exception(type->underlying_type(), nullptr) {}

MemoryError::MemoryError(PyTuple *args) : Exception(types::BuiltinTypes::the().memory_error(), args)
{}

PyResult<PyObject *> MemoryError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::memory_error());
	ASSERT(!kwargs || kwargs->map().empty());
	if (auto result = MemoryError::create(args); result.is_ok()) {
		return Ok(static_cast<PyObject *>(result.unwrap()));
	} else {
		return Err(result.unwrap_err());
	}
}

PyType *MemoryError::static_type() const
{
	ASSERT(types::memory_error());
	return types::memory_error();
}

PyType *MemoryError::this_type()
{
	ASSERT(types::memory_error());
	return types::memory_error();
}

std::string MemoryError::to_string() const { return what(); }

namespace {

	std::once_flag memory_error_flag;

	std::unique_ptr<TypePrototype> register_memory_error()
	{
		return std::move(klass<MemoryError>("MemoryError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> MemoryError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(memory_error_flag, []() { type = register_memory_error(); });
		return std::move(type);
	};
}

}// namespace py
