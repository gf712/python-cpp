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
		auto *current_frame = interpreter.execution_frame();
		interpreter.set_execution_frame(current_frame->exit());
		spdlog::debug("Manually deallocationg ExecutionFrame {}", (void *)current_frame);
		// TODO: make deallocation more developer friendly
		current_frame->~ExecutionFrame();
		reinterpret_cast<GarbageCollected *>((uint8_t *)current_frame - sizeof(GarbageCollected))
			->mark(GarbageCollected::Color::WHITE);
	}

	vm.reg(0) = result;
}
