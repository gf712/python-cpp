#include "LoadMethod.hpp"


void LoadMethod::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto this_value = vm.reg(m_value_source);
	spdlog::debug("This object: {}",
		std::visit([](const auto &val) { return PyObject::from(val)->to_string(); }, this_value));
	if (auto *this_obj = std::get_if<std::shared_ptr<PyObject>>(&this_value)) {
        auto maybe_method = (*this_obj)->get(m_method_name, interpreter);
        ASSERT(maybe_method->type() == PyObjectType::PY_FUNCTION)
		vm.reg(m_destination) = std::move(maybe_method);
	} else {
		TODO();
	}
}
