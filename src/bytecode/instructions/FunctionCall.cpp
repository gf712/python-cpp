#include "FunctionCall.hpp"

void FunctionCall::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto function_frame = ExecutionFrame::create(interpreter.execution_frame());

	// dummy for RAII, so that when we exit function we pop the stack

	auto function_object = std::get<std::shared_ptr<PyObject>>(vm.reg(m_function_name));

	if (auto pyfunc = as<PyFunction>(function_object)) {
		const auto offset = pyfunc->code()->offset();
		interpreter.set_execution_frame(function_frame);

		for (size_t i = 0; i < m_args.size(); ++i) {
			function_frame->parameter(i) = vm.reg(m_args[i]);
		}
		// std::visit([](const auto &val) { std::cout << val << '\n'; },
		// function_frame->parameter(0));
		function_frame->set_return_address(vm.instruction_pointer());
		vm.set_instruction_pointer(offset);
		auto frame = vm.enter_frame(pyfunc->code()->register_count());
		function_frame->attach_frame(std::move(frame));
	} else if (auto native_func = as<PyNativeFunction>(function_object)) {
		auto obj =
			std::visit([](const auto &value) { return PyObject::from(value); }, vm.reg(m_args[0]));
		vm.reg(0) = native_func->operator()(obj);
	} else {
		TODO();
	}
}