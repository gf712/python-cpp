#include "StoreAttr.hpp"
#include "runtime/PyString.hpp"

using namespace py;

PyResult StoreAttr::execute(VirtualMachine &vm, Interpreter &) const
{
	auto this_value = vm.reg(m_dst);
	spdlog::debug("This object: {}",
		std::visit(
			[](const auto &val) {
				auto obj = PyObject::from(val);
				ASSERT(obj.is_ok())
				return obj.template unwrap_as<PyObject>()->to_string();
			},
			this_value));
	if (auto *this_obj = std::get_if<PyObject *>(&this_value)) {
		auto other_obj =
			std::visit([](const auto &val) { return PyObject::from(val); }, vm.reg(m_src));
		if (other_obj.is_err()) return other_obj;
		auto attr_name = PyString::create(m_attr_name);
		if (attr_name.is_err()) { return attr_name; }
		return (*this_obj)->setattribute(
			attr_name.unwrap_as<PyString>(), other_obj.unwrap_as<PyObject>());
	} else {
		TODO();
		return PyResult::Err(nullptr);
	}
}

std::vector<uint8_t> StoreAttr::serialize() const
{
	TODO();
	return {
		STORE_ATTR,
		m_dst,
		m_src,
	};
}
