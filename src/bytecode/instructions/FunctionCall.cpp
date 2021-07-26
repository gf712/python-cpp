#include "FunctionCall.hpp"

std::shared_ptr<PyObject> execute(VirtualMachine &vm,
	Interpreter &interpreter,
	std::shared_ptr<PyObject> function_object,
	const std::shared_ptr<PyTuple> &args)
{
	std::string function_name;

	if (auto pyfunc = as<PyFunction>(function_object)) {
		function_name = pyfunc->name();
	} else if (auto native_func = as<PyNativeFunction>(function_object)) {
		function_name = native_func->name();
	} else {
		TODO();
	}

	auto scope_name = interpreter.execution_frame()->fetch_object("__name__");
	ASSERT(scope_name->type() == PyObjectType::PY_STRING)
	if (auto pyfunc = as<PyFunction>(function_object)) {
		function_name = fmt::format("{}.{}", as<PyString>(scope_name)->value(), function_name);
		auto function_frame = ExecutionFrame::create(interpreter.execution_frame(), function_name);
		const auto offset = pyfunc->code()->offset();
		interpreter.set_execution_frame(function_frame);

		size_t i = 0;
		for (const auto &arg : *args) { function_frame->parameter(i++) = arg; }
		// std::visit([](const auto &val) { std::cout << val << '\n'; },
		// function_frame->parameter(0));
		function_frame->set_return_address(vm.instruction_pointer());
		vm.set_instruction_pointer(offset);

		// function stack that uses RAII, so that when we exit function we pop the stack
		// also allocates virtual registers needed by this function
		auto frame = vm.enter_frame(pyfunc->code()->register_count());
		function_frame->attach_frame(std::move(frame));
	} else if (auto native_func = as<PyNativeFunction>(function_object)) {
		auto result = native_func->operator()(args);
		vm.reg(0) = result;
	} else {
		TODO();
	}
	return nullptr;
}

void FunctionCall::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto func = vm.reg(m_function_name);
	ASSERT(std::get_if<std::shared_ptr<PyObject>>(&func));
	auto function_object = std::get<std::shared_ptr<PyObject>>(func);

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }

	auto args_tuple = vm.heap().allocate<PyTuple>(args);

	ASSERT(args_tuple);

	::execute(vm, interpreter, function_object, args_tuple);
}