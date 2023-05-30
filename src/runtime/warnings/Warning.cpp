#include "Warning.hpp"
#include "runtime/PyString.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"

namespace py {

Warning::Warning(PyType *type) : Exception(type) {}

Warning::Warning(PyType *, PyTuple *args) : Exception(types::BuiltinTypes::the().warning(), args) {}

PyResult<Warning *> Warning::create(PyType *type, PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<Warning>(type, args);
	if (!result) { return Err(memory_error(sizeof(Warning))); }
	return Ok(result);
}

PyResult<PyObject *> Warning::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty())
	return Warning::create(const_cast<PyType *>(type), args);
}

PyType *Warning::static_type() const
{
	ASSERT(types::warning());
	return types::warning();
}

PyType *Warning::class_type()
{
	ASSERT(types::warning());
	return types::warning();
}

namespace {

	std::once_flag warning_flag;

	std::unique_ptr<TypePrototype> register_warning()
	{
		return std::move(klass<Warning>("Warning", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> Warning::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(warning_flag, []() { type = register_warning(); });
		return std::move(type);
	};
}

}// namespace py
