#include "NotImplemented.hpp"
#include "MemoryError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

NotImplemented::NotImplemented(PyType *type) : PyBaseObject(type) {}

NotImplemented::NotImplemented() : PyBaseObject(BuiltinTypes::the().not_implemented_()) {}

PyResult<NotImplemented *> NotImplemented::create()
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate_static<NotImplemented>().get();
	if (!result) { return Err(memory_error(sizeof(NotImplemented))); }
	return Ok(result);
}

PyType *NotImplemented::static_type() const { return not_implemented_(); }

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

NotImplemented *not_implemented()
{
	static NotImplemented *value = nullptr;
	if (!value) { value = NotImplemented::create().unwrap(); }
	return value;
}

}// namespace py
