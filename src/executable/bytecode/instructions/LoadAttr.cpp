#include "LoadAttr.hpp"
#include "runtime/PyModule.hpp"

void LoadAttr::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto this_value = vm.reg(m_value_source);
	spdlog::debug("This object: {}",
		std::visit([](const auto &val) { return PyObject::from(val)->to_string(); }, this_value));
	if (auto *this_obj = std::get_if<PyObject *>(&this_value)) {
		if (auto module = as<PyModule>(*this_obj)) {
			vm.reg(m_destination) = module->symbol_table().at(PyString::create(m_attr_name));
		} else {
			vm.reg(m_destination) = (*this_obj)->get(m_attr_name, interpreter);
		}
	} else {
		TODO();
	}
}
