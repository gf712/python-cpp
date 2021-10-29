#include "FunctionCall.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyString.hpp"
#include "runtime/TypeError.hpp"

PyObject *execute(VirtualMachine &vm,
	Interpreter &interpreter,
	PyObject *function_object,
	PyTuple *args,
	PyDict *kwargs,
	PyDict *ns)
{
	std::string function_name;

	if (auto pyfunc = as<PyFunction>(function_object)) {
		function_name = pyfunc->name();
	} else if (auto native_func = as<PyNativeFunction>(function_object)) {
		function_name = native_func->name();
	} else {
		TODO();
	}

	if (auto pyfunc = as<PyFunction>(function_object)) {
		// FIXME: this scope makes sure that the local function_frame object is destroyed
		// 		  before the VM is executes the frame. Not sure if ExecutionFrame could be
		//		  std::unique_ptr (might be access from multiple threads later on?)
		{
			auto function_locals = VirtualMachine::the().heap().allocate<PyDict>();
			auto function_frame = ExecutionFrame::create(interpreter.execution_frame(),
				interpreter.execution_frame()->globals(),
				function_locals,
				ns);
			const auto offset = pyfunc->code()->offset();
			interpreter.set_execution_frame(function_frame);

			size_t i = 0;
			if (args) {
				for (const auto &arg : *args) { function_frame->parameter(i++) = arg; }
			}
			if (kwargs) {
				const auto &argnames = pyfunc->code()->args();
				for (const auto &[key, value] : kwargs->map()) {
					ASSERT(std::holds_alternative<String>(key))
					auto key_str = std::get<String>(key);
					auto arg_iter = std::find(argnames.begin(), argnames.end(), key_str.s);
					if (arg_iter == argnames.end()) {
						type_error("{}() got an unexpected keyword argument '{}'",
							function_name,
							key_str.s);
						return nullptr;
					}
					auto &arg =
						function_frame->parameter(std::distance(argnames.begin(), arg_iter));
					if (arg.has_value()) {
						type_error(
							"{}() got multiple values for argument '{}'", function_name, key_str.s);
						return nullptr;
					}
					arg = value;
				}
			}

			function_frame->set_return_address(vm.instruction_pointer());
			vm.set_instruction_pointer(offset);

			// function stack that uses RAII, so that when we exit function we pop the stack
			// also allocates virtual registers needed by this function
			spdlog::debug("Requesting stack frame with {} virtual registers",
				pyfunc->code()->register_count());
			auto frame = vm.enter_frame(pyfunc->code()->register_count());
			function_frame->attach_frame(std::move(frame));
		}
		const auto &execution_frame = interpreter.execution_frame();
		vm.execute_frame();

		spdlog::debug("Frame: {}", (void *)execution_frame.get());
		spdlog::debug("Locals: {}", execution_frame->locals()->to_string());
		spdlog::debug("Globals: {}", execution_frame->globals()->to_string());
		if (ns) { spdlog::info("Namespace: {}", ns->to_string()); }
	} else if (auto native_func = as<PyNativeFunction>(function_object)) {
		auto result = native_func->operator()(args, kwargs);
		spdlog::debug("Native function return value: {}", result->to_string());
		vm.reg(0) = result;
	} else {
		TODO();
	}
	return std::visit([](const auto &val) { return PyObject::from(val); }, vm.reg(0));
}

void FunctionCall::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto func = vm.reg(m_function_name);
	ASSERT(std::get_if<PyObject *>(&func));
	auto function_object = std::get<PyObject *>(func);

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }

	auto *args_tuple = PyTuple::create(args);

	ASSERT(args_tuple);

	::execute(vm, interpreter, function_object, args_tuple, nullptr, nullptr);
}