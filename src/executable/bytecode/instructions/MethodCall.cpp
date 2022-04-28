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

PyResult MethodCall::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto &maybe_self = vm.reg(m_caller);
	auto *obj = std::get<PyObject *>(maybe_self);
	ASSERT(obj)

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }

	auto args_tuple = PyTuple::create(args);
	if (args_tuple.is_err()) { return args_tuple; }

	// FIXME: process kwargs
	PyDict *kwargs = nullptr;

	auto result = obj->call(args_tuple.unwrap_as<PyTuple>(), kwargs);
	if (result.is_ok()) { vm.reg(0) = result.unwrap(); }
	return result;
}

std::vector<uint8_t> MethodCall::serialize() const
{
	TODO();
	return {
		METHOD_CALL,
		m_caller,
	};
}