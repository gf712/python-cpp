#include "NameError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

NameError::NameError(PyType *type) : Exception(type) {}

NameError::NameError(PyTuple *args) : Exception(types::BuiltinTypes::the().name_error(), args) {}

PyType *NameError::static_type() const
{
	ASSERT(types::name_error());
	return types::name_error();
}

namespace {

	std::once_flag name_error_flag;

	std::unique_ptr<TypePrototype> register_name_error()
	{
		return std::move(klass<NameError>("NameError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> NameError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(name_error_flag, []() { type = register_name_error(); });
		return std::move(type);
	};
}
}// namespace py
