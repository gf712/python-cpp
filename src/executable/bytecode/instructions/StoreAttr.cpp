#include "StoreAttr.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyString.hpp"

using namespace py;

PyResult<Value> StoreAttr::execute(VirtualMachine &vm, Interpreter &) const
{
	auto this_value = vm.reg(m_dst);
	spdlog::debug("This object: {}",
		std::visit(
			[](const auto &val) {
				auto obj = PyObject::from(val);
				ASSERT(obj.is_ok())
				return obj.unwrap()->to_string();
			},
			this_value));
	if (auto *this_obj = std::get_if<PyObject *>(&this_value)) {
		auto other_obj =
			std::visit([](const auto &val) { return PyObject::from(val); }, vm.reg(m_src));
		if (other_obj.is_err()) return Err(other_obj.unwrap_err());
		auto attr_name = PyString::create(m_attr_name);
		if (attr_name.is_err()) { return Err(attr_name.unwrap_err()); }
		if (auto result = (*this_obj)->setattribute(attr_name.unwrap(), other_obj.unwrap());
			result.is_ok()) {
			return Ok(py_none());
		} else {
			return Err(result.unwrap_err());
		}
	} else {
		TODO();
		return Err(nullptr);
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
