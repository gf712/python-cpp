#include "NotImplemented.hpp"

#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

NotImplemented::NotImplemented(PyType *type) : PyBaseObject(type) {}

NotImplemented::NotImplemented() : PyBaseObject(BuiltinTypes::the().not_implemented()) {}

PyType *NotImplemented::static_type() const { return not_implemented(); }

std::string NotImplemented::to_string() const { return "NotImplemented"; }

namespace {
	std::once_flag not_implemented_flag;

	std::unique_ptr<TypePrototype> register_not_implemented()
	{
		return std::move(klass<NotImplemented>("NotImplemented").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> NotImplemented::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(not_implemented_flag, []() { type = register_not_implemented(); });
		return std::move(type);
	};
}

}// namespace py
