#include "RaiseVarargs.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyTuple.hpp"

void RaiseVarargs::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto &assertion_function = vm.reg(m_assertion);
	ASSERT(std::holds_alternative<PyObject *>(assertion_function))
	auto *obj = std::get<PyObject *>(assertion_function);
	ASSERT(as<PyNativeFunction>(obj))

	std::vector<Value> args;
	args.reserve(m_args.size());
	for (const auto &arg : m_args) { args.push_back(vm.reg(arg)); }

	as<PyNativeFunction>(obj)->operator()(PyTuple::create(args), nullptr);
}