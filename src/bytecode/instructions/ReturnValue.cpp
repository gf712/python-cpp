#include "ReturnValue.hpp"


void ReturnValue::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto result = vm.reg(m_source);

	std::visit(
		overloaded{ [](const auto &val) {
					   std::ostringstream os;
					   os << val;
					   spdlog::debug("Return value: {}", os.str());
				   },
			[](const PyObject *val) { spdlog::debug("Return value: {}", val->to_string()); } },
		result);
	if (interpreter.execution_frame()->parent()) {
		vm.set_instruction_pointer(interpreter.execution_frame()->return_address());
		auto current_frame = std::unique_ptr<ExecutionFrame, void (*)(ExecutionFrame *)>(
			interpreter.execution_frame(), [](ExecutionFrame *ptr) {
				spdlog::debug("Deallocationg ExecutionFrame {}", (void *)ptr);
				ptr->~ExecutionFrame();
			});
		interpreter.set_execution_frame(current_frame->exit());
	}

	vm.reg(0) = result;
}
