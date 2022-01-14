#include "StoreAttr.hpp"
#include "runtime/PyString.hpp"

using namespace py;

void StoreAttr::execute(VirtualMachine &vm, Interpreter &) const
{
	auto this_value = vm.reg(m_dst);
	spdlog::debug("This object: {}",
		std::visit([](const auto &val) { return PyObject::from(val)->to_string(); }, this_value));
	if (auto *this_obj = std::get_if<PyObject *>(&this_value)) {
		auto other_obj =
			std::visit([](const auto &val) { return PyObject::from(val); }, vm.reg(m_src));
		(*this_obj)->setattribute(PyString::create(m_attr_name), other_obj);
	} else {
		TODO();
	}
}
