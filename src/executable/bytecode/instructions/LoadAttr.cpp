#include "LoadAttr.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"

using namespace py;

PyResult LoadAttr::execute(VirtualMachine &vm, Interpreter &) const
{
	auto this_value = vm.reg(m_value_source);
	spdlog::debug("This object: {}",
		std::visit(
			[](const auto &val) {
				auto obj = PyObject::from(val);
				ASSERT(obj.is_ok())
				return obj.template unwrap_as<PyObject>()->to_string();
			},
			this_value));
	auto result = [&]() {
		if (auto *this_obj = std::get_if<PyObject *>(&this_value)) {
			// FIXME: is it worth unifying this?
			if (auto module = as<PyModule>(*this_obj)) {
				auto name = PyString::create(m_attr_name);
				if (name.is_err()) { return name; }
				return PyResult::Ok(module->symbol_table().at(name.unwrap_as<PyString>()));
			} else {
				auto name = PyString::create(m_attr_name);
				return (*this_obj)->get_attribute(name.unwrap_as<PyString>());
			}
		} else {
			TODO();
			return PyResult::Err(nullptr);
		}
	}();

	if (result.is_ok()) { vm.reg(m_destination) = result.unwrap(); }
	return result;
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