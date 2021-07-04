#include "FunctionCall.hpp"

void FunctionCall::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto function_frame = ExecutionFrame::create(interpreter.execution_frame());

	auto function_object = std::get<std::shared_ptr<PyObject>>(vm.reg(m_function_name));

	if (auto pyfunc = as<PyFunction>(function_object)) {
		const auto offset = pyfunc->code()->offset();

		interpreter.set_execution_frame(function_frame);

		for (size_t i = 0; i < m_args.size(); ++i) {
            function_frame->parameter(i) = vm.reg(m_args[i]);
		}

		vm.set_return_address(vm.instruction_pointer());
		vm.set_instruction_pointer(offset);
	} else if (auto native_func = as<PyNativeFunction>(function_object)) {
		vm.reg(0) = native_func->operator()(std::get<std::shared_ptr<PyObject>>(vm.reg(m_args[0])));
	} else {
		TODO();
	}
}