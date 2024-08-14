#include "runtime/warnings/ResourceWarning.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"
#include "runtime/warnings/Warning.hpp"

using namespace py;

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
	ASSERT(!kwargs || kwargs->map().empty())
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
