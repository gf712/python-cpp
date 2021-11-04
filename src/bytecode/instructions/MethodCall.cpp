#include "FunctionCall.hpp"
#include "MethodCall.hpp"

#include "runtime/PyDict.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"


void MethodCall::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto func = vm.reg(m_function_name);
	ASSERT(std::get_if<PyObject *>(&func));
	auto *function_object = std::get<PyObject *>(func);

	std::vector<Value> args;
	// self
	args.push_back(interpreter.execution_frame()->locals()->map().at(
		PyString::from(String{ m_instance_name })));
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }

	auto* args_tuple = PyTuple::create(args);

	ASSERT(args_tuple);

	// TODO: add support for kwargs
	::execute(vm, interpreter, function_object, args_tuple, nullptr, nullptr);
}