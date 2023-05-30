#include "LookupError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

LookupError::LookupError(PyType *type) : Exception(type->underlying_type(), nullptr) {}

LookupError::LookupError(PyType *type, PyTuple *args) : Exception(type->underlying_type(), args) {}

LookupError::LookupError(TypePrototype &type, PyTuple *args) : Exception(type, args) {}

LookupError::LookupError(PyTuple *args) : Exception(types::BuiltinTypes::the().lookup_error(), args)
{}

PyResult<PyObject *> LookupError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::lookup_error());
	ASSERT(!kwargs || kwargs->map().empty())
	if (auto result = LookupError::create(args)) {
		return Ok(static_cast<PyObject *>(result));
	} else {
		TODO();
	}
}

PyType *LookupError::class_type()
{
	ASSERT(types::lookup_error());
	return types::lookup_error();
}

PyType *LookupError::static_type() const
{
	ASSERT(types::lookup_error());
	return types::lookup_error();
}

namespace {

	std::once_flag lookup_error_flag;

	std::unique_ptr<TypePrototype> register_lookup_error()
	{
		return std::move(klass<LookupError>("LookupError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> LookupError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(lookup_error_flag, []() { type = register_lookup_error(); });
		return std::move(type);
	};
}
}// namespace py
