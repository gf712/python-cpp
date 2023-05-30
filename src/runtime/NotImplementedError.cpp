#include "NotImplementedError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

NotImplementedError::NotImplementedError(PyType *type) : Exception(type) {}

NotImplementedError::NotImplementedError(PyTuple *args)
	: Exception(types::BuiltinTypes::the().not_implemented_error(), args)
{}

PyResult<PyObject *> NotImplementedError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::not_implemented_error());
	ASSERT(!kwargs || kwargs->map().empty())
	if (auto result = NotImplementedError::create(args)) {
		return Ok(static_cast<PyObject *>(result));
	} else {
		TODO();
	}
}

PyType *NotImplementedError::static_type() const
{
	ASSERT(types::not_implemented_error());
	return types::not_implemented_error();
}

namespace {

	std::once_flag not_implemented_error_flag;

	std::unique_ptr<TypePrototype> register_not_implemented_error()
	{
		return std::move(
			klass<NotImplementedError>("NotImplementedError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> NotImplementedError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(
			not_implemented_error_flag, []() { type = register_not_implemented_error(); });
		return std::move(type);
	};
}
}// namespace py
