module;
#include "core.hpp"
#include "memory/allocate.hpp"

module py.runtime;
import py.types;


using namespace py;

// Declared in ResourceWarning.cppm. In namespace py rather than under the
// using-directive above, which would define an unrelated function at global scope.
namespace py {
BaseException *make_resource_warning(std::string &&message)
{
	auto msg = PyString::create(std::move(message));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return ResourceWarning::create(ResourceWarning::class_type(), args_tuple.unwrap()).unwrap();
}
}// namespace py

ResourceWarning::ResourceWarning(PyType *type) : Warning(type) {}

ResourceWarning::ResourceWarning(PyType *, PyTuple *args) : Warning(types::resource_warning(), args)
{}

PyResult<ResourceWarning *> ResourceWarning::create(PyType *type, PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<ResourceWarning>(type, args);
	if (!result) { return Err(memory_error(sizeof(Warning))); }
	return Ok(result);
}

PyResult<PyObject *> ResourceWarning::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty());
	return ResourceWarning::create(const_cast<PyType *>(type), args);
}

PyType *ResourceWarning::static_type() const
{
	ASSERT(types::resource_warning());
	return types::resource_warning();
}

PyType *ResourceWarning::class_type()
{
	ASSERT(types::resource_warning());
	return types::resource_warning();
}

namespace {

std::once_flag resource_warning_flag;

std::unique_ptr<TypePrototype> register_resource_warning()
{
	return std::move(klass<ResourceWarning>("ResourceWarning", types::warning()).type);
}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> ResourceWarning::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(resource_warning_flag, []() { type = register_resource_warning(); });
		return std::move(type);
	};
}
