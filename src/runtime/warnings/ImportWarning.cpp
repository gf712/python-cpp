#include "ImportWarning.hpp"
#include "runtime/PyString.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"

namespace py {

ImportWarning::ImportWarning(PyType *type) : Warning(type) {}

ImportWarning::ImportWarning(PyType *type, PyTuple *args) : Warning(type, args) {}

PyResult<ImportWarning *> ImportWarning::create(PyType *type, PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<ImportWarning>(type, args);
	if (!result) { return Err(memory_error(sizeof(ImportWarning))); }
	return Ok(result);
}

PyResult<PyObject *> ImportWarning::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty());
	return ImportWarning::create(const_cast<PyType *>(type), args);
}

PyType *ImportWarning::static_type() const
{
	ASSERT(types::import_warning());
	return types::import_warning();
}

PyType *ImportWarning::class_type()
{
	ASSERT(types::import_warning());
	return types::import_warning();
}

namespace {

	std::once_flag import_warning_flag;

	std::unique_ptr<TypePrototype> register_import_warning()
	{
		return std::move(klass<ImportWarning>("ImportWarning", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> ImportWarning::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(import_warning_flag, []() { type = register_import_warning(); });
		return std::move(type);
	};
}

}// namespace py
