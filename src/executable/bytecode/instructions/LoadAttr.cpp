#include "LoadAttr.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> LoadAttr::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto this_value = vm.reg(m_value_source);
	const auto &attribute_name = interpreter.execution_frame()->names(m_attr_name);
	spdlog::debug("This object: {}",
		std::visit(
			[](const auto &val) {
				auto obj = PyObject::from(val);
				ASSERT(obj.is_ok())
				return obj.unwrap()->to_string();
			},
			this_value));
	auto result = [&]() -> PyResult<Value> {
		if (auto *this_obj = std::get_if<PyObject *>(&this_value)) {
			auto name = PyString::create(attribute_name);
			if (auto r = (*this_obj)->get_attribute(name.unwrap()); r.is_ok()) {
				return Ok(Value{ r.unwrap() });
			} else {
				return Err(r.unwrap_err());
			}
		} else {
			auto result = PyObject::from(this_value).and_then([&attribute_name](PyObject *obj) {
				auto name = PyString::create(attribute_name);
				return obj->get_attribute(name.unwrap());
			});
			if (result.is_ok()) {
				return Ok(Value{ result.unwrap() });
			} else {
				return Err(result.unwrap_err());
			}
		}
	}();

	if (result.is_ok()) { vm.reg(m_destination) = result.unwrap(); }
	return result;
}

std::vector<uint8_t> LoadAttr::serialize() const
{
	return {
		LOAD_ATTR,
		m_destination,
		m_value_source,
		m_attr_name,
	};
}