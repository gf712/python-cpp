#include "LoadAttr.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"

using namespace py;

void LoadAttr::execute(VirtualMachine &vm, Interpreter &) const
{
	auto this_value = vm.reg(m_value_source);
	spdlog::debug("This object: {}",
		std::visit([](const auto &val) { return PyObject::from(val)->to_string(); }, this_value));
	if (auto *this_obj = std::get_if<PyObject *>(&this_value)) {
		// FIXME: is it worth unifying this?
		if (auto module = as<PyModule>(*this_obj)) {
			auto *name = PyString::create(m_attr_name);
			vm.reg(m_destination) = module->symbol_table().at(name);
		} else {
			vm.reg(m_destination) = (*this_obj)->get_attribute(PyString::create(m_attr_name));
		}
	} else {
		TODO();
	}
}

std::vector<uint8_t> LoadAttr::serialize() const
{
	// attribute name has to be loaded from the stack
	TODO();
	return {
		LOAD_ATTR,
		m_destination,
		m_value_source,
	};
}