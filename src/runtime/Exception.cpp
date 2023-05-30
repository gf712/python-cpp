#include "Exception.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

Exception::Exception(PyType *t) : BaseException(t->underlying_type(), nullptr) {}

Exception::Exception(PyType *t, PyTuple *args) : BaseException(t, args) {}

Exception::Exception(PyTuple *args) : BaseException(types::BuiltinTypes::the().exception(), args) {}

Exception::Exception(const TypePrototype &type, PyTuple *args) : BaseException(type, args) {}

PyType *Exception::static_type() const
{
	ASSERT(types::exception());
	return types::exception();
}

PyType *Exception::class_type()
{
	ASSERT(types::exception());
	return types::exception();
}

namespace {

	std::once_flag exception_flag;

	std::unique_ptr<TypePrototype> register_exception()
	{
		return std::move(klass<Exception>("Exception", BaseException::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> Exception::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(exception_flag, []() { type = register_exception(); });
		return std::move(type);
	};
}
}// namespace py
