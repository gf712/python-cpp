#include "runtime/warnings/PendingDeprecationWarning.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"
#include "runtime/warnings/Warning.hpp"

using namespace py;

PendingDeprecationWarning::PendingDeprecationWarning(PyType *type) : Warning(type) {}

PendingDeprecationWarning::PendingDeprecationWarning(PyType *, PyTuple *args)
	: Warning(types::pending_deprecation_warning(), args)
{}

PyResult<PendingDeprecationWarning *> PendingDeprecationWarning::create(PyType *type, PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<PendingDeprecationWarning>(type, args);
	if (!result) { return Err(memory_error(sizeof(Warning))); }
	return Ok(result);
}

PyResult<PyObject *>
	PendingDeprecationWarning::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty());
	return PendingDeprecationWarning::create(const_cast<PyType *>(type), args);
}

PyType *PendingDeprecationWarning::static_type() const
{
	ASSERT(types::pending_deprecation_warning());
	return types::pending_deprecation_warning();
}

PyType *PendingDeprecationWarning::class_type()
{
	ASSERT(types::pending_deprecation_warning());
	return types::pending_deprecation_warning();
}

namespace {

std::once_flag pending_deprecation_warning_flag;

std::unique_ptr<TypePrototype> register_pending_deprecation_warning()
{
	return std::move(
		klass<PendingDeprecationWarning>("PendingDeprecationWarning", types::warning()).type);
}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PendingDeprecationWarning::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(pending_deprecation_warning_flag,
			[]() { type = register_pending_deprecation_warning(); });
		return std::move(type);
	};
}
