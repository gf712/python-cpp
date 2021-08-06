#include "ReturnValue.hpp"


void ReturnValue::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto result = vm.reg(m_source);

	std::visit(overloaded{ [](const auto &val) {
							  std::ostringstream os;
							  os << val;
							  spdlog::debug("Return value: {}", os.str());
						  },
				   [](const std::shared_ptr<PyObject> &val) {
					   spdlog::debug("Return value: {}", val->to_string());
				   } },
		result);
	if (interpreter.execution_frame()->parent()) {
		vm.set_instruction_pointer(interpreter.execution_frame()->return_address());
		interpreter.set_execution_frame(interpreter.execution_frame()->pop());
	}

	vm.reg(0) = result;
}
