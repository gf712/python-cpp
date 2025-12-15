#include "DeleteAttr.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyString.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> DeleteAttr::execute(VirtualMachine &vm, Interpreter &intepreter) const
{
	auto this_value = vm.reg(m_self);
	const auto &attr_name_ = intepreter.execution_frame()->names(m_attr_name);
	spdlog::debug("This object: {}",
		std::visit(
			[](const auto &val) {
				auto obj = PyObject::from(val);
				ASSERT(obj.is_ok());
				return obj.unwrap()->to_string();
			},
			this_value));
	if (auto *this_obj = std::get_if<PyObject *>(&this_value)) {
		auto attr_name = PyString::create(attr_name_);
		if (attr_name.is_err()) { return Err(attr_name.unwrap_err()); }
		[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
		if (auto result = (*this_obj)->delattribute(attr_name.unwrap()); result.is_ok()) {
			return Ok(py_none());
		} else {
			return Err(result.unwrap_err());
		}
	} else {
		TODO();
		return Err(nullptr);
	}
}

std::vector<uint8_t> DeleteAttr::serialize() const
{
	return {
		DELETE_ATTR,
		m_self,
		m_attr_name,
	};
}
