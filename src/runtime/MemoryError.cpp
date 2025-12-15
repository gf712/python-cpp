#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

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
