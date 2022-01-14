#include "MethodCall.hpp"
#include "FunctionCall.hpp"

#include "runtime/AttributeError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyMethodDescriptor.hpp"
#include "runtime/PySlotWrapper.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"

using namespace py;

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

	if (auto *result = obj->call(args_tuple, kwargs)) { vm.reg(0) = result; }
}