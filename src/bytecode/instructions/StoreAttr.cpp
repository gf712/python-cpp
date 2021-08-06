#include "StoreAttr.hpp"


void StoreAttr::execute(VirtualMachine &vm, Interpreter &) const
{
	auto this_value = vm.reg(m_dst);
	spdlog::debug("This object: {}",
		std::visit([](const auto &val) { return PyObject::from(val)->to_string(); }, this_value));
	if (auto *this_obj = std::get_if<std::shared_ptr<PyObject>>(&this_value)) {
		auto other_obj =
			std::visit([](const auto &val) { return PyObject::from(val); }, vm.reg(m_src));
		(*this_obj)->put(m_attr_name, other_obj);
	} else {
		TODO();
	}
}
