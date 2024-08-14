#include "DeprecationWarning.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"
#include "runtime/warnings/Warning.hpp"

using namespace py;

DeprecationWarning::DeprecationWarning(PyType *type) : Warning(type) {}

DeprecationWarning::DeprecationWarning(PyType *, PyTuple *args)
	: Warning(types::deprecation_warning(), args)
{}

PyResult<DeprecationWarning *> DeprecationWarning::create(PyType *type, PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<DeprecationWarning>(type, args);
	if (!result) { return Err(memory_error(sizeof(Warning))); }
	return Ok(result);
}

PyResult<PyObject *> DeprecationWarning::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty())
	return DeprecationWarning::create(const_cast<PyType *>(type), args);
}

PyType *DeprecationWarning::static_type() const
{
	ASSERT(types::deprecation_warning());
	return types::deprecation_warning();
}

PyType *DeprecationWarning::class_type()
{
	ASSERT(types::deprecation_warning());
	return types::deprecation_warning();
}

namespace {

std::once_flag deprecation_warning_flag;

std::unique_ptr<TypePrototype> register_deprecation_warning()
{
	return std::move(klass<DeprecationWarning>("DeprecationWarning", types::warning()).type);
}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> DeprecationWarning::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(deprecation_warning_flag, []() { type = register_deprecation_warning(); });
		return std::move(type);
	};
}
