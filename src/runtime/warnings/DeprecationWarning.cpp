module;
#include "core.hpp"
#include "memory/allocate.hpp"

module py.runtime;
import py.types;


using namespace py;

// Declared in DeprecationWarning.cppm. In namespace py rather than under the
// using-directive above, which would define an unrelated function at global scope.
namespace py {
BaseException *make_deprecation_warning(std::string &&message)
{
	auto msg = PyString::create(std::move(message));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return DeprecationWarning::create(DeprecationWarning::class_type(), args_tuple.unwrap())
		.unwrap();
}
}// namespace py

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
	ASSERT(!kwargs || kwargs->map().empty());
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
