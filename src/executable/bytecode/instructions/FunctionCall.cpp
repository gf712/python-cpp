#include "FunctionCall.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyTuple.hpp"


void FunctionCall::execute(VirtualMachine &vm, Interpreter &) const
{
	auto func = vm.reg(m_function_name);
	ASSERT(std::get_if<PyObject *>(&func));
	auto callable_object = std::get<PyObject *>(func);

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }

	auto *args_tuple = PyTuple::create(args);
	spdlog::debug("args_tuple: {}", (void *)&args_tuple);
	ASSERT(args_tuple);

	vm.reg(0) = callable_object->call(args_tuple, nullptr);
}