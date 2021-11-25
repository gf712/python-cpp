#include "BaseException.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

BaseException::BaseException(std::string exception_name, std::string &&name)
	: PyBaseObject(PyObjectType::PY_BASE_EXCEPTION, BuiltinTypes::the().exception()),
	  m_exception_name(std::move(exception_name)), m_message(std::move(name))
{}

PyType *BaseException::type_() const { return exception(); }

namespace {

std::once_flag base_exception_flag;

std::unique_ptr<TypePrototype> register_base_exception()
{
	return std::move(klass<BaseException>("exception").type);
}
}// namespace

std::unique_ptr<TypePrototype> BaseException::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(base_exception_flag, []() { type = ::register_base_exception(); });
	return std::move(type);
}