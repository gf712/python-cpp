#include "OSError.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "runtime/PyType.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

template<> OSError *as(PyObject *obj)
{
	ASSERT(types::os_error());
	if (obj->type() == types::os_error()) { return static_cast<OSError *>(obj); }
	return nullptr;
}

template<> const OSError *as(const PyObject *obj)
{
	ASSERT(types::os_error());
	if (obj->type() == types::os_error()) { return static_cast<const OSError *>(obj); }
	return nullptr;
}

OSError::OSError(PyType *type, PyTuple *args) : Exception(type->underlying_type(), args) {}

OSError::OSError(PyTuple *args) : Exception(types::BuiltinTypes::the().os_error(), args) {}

OSError::OSError(PyType *t) : OSError(t, nullptr) {}

PyResult<OSError *> OSError::create(PyType *type, PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto result = heap.allocate<OSError>(type, args);
	if (!result) { return Err(memory_error(sizeof(OSError))); }
	return Ok(result);
}

PyResult<OSError *> OSError::create(PyTuple *args) { return create(types::os_error(), args); }

PyResult<PyObject *> OSError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty());
	return OSError::create(const_cast<PyType *>(type), args);
}

PyType *OSError::static_type() const
{
	ASSERT(types::os_error());
	return types::os_error();
}

PyType *OSError::class_type()
{
	ASSERT(types::os_error());
	return types::os_error();
}

namespace {

	std::once_flag os_error_flag;

	std::unique_ptr<TypePrototype> register_os_error()
	{
		return std::move(klass<OSError>("OSError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> OSError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(os_error_flag, []() { type = register_os_error(); });
		return std::move(type);
	};
}
}// namespace py
