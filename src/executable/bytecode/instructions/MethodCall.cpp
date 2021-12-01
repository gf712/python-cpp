#include "MethodCall.hpp"
#include "FunctionCall.hpp"

#include "runtime/AttributeError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyMethodWrapper.hpp"
#include "runtime/PySlotWrapper.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"


void MethodCall::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto &maybe_self = vm.reg(m_caller);
	auto *obj = std::get<PyObject *>(maybe_self);
	ASSERT(obj)

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }

	auto *args_tuple = PyTuple::create(args);

	// FIXME: process kwargs
	PyDict *kwargs = nullptr;

	vm.reg(0) = obj->call(args_tuple, kwargs);
}