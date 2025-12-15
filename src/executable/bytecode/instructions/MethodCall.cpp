#include "MethodCall.hpp"
#include "FunctionCall.hpp"

#include "runtime/AttributeError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyMethodDescriptor.hpp"
#include "runtime/PySlotWrapper.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"
#include "vm/VM.hpp"

#include "../serialization/serialize.hpp"

using namespace py;

PyResult<Value> MethodCall::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto &method = vm.reg(m_caller);
	auto *method_obj = std::get<PyObject *>(method);
	ASSERT(method_obj);

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }

	auto args_tuple = PyTuple::create(args);
	if (args_tuple.is_err()) { return Err(args_tuple.unwrap_err()); }

	// FIXME: process kwargs
	PyDict *kwargs = nullptr;

	spdlog::debug("calling method: \'{}\'", method_obj->to_string());

	auto result = method_obj->call(args_tuple.unwrap(), kwargs);
	if (result.is_err()) return Err(result.unwrap_err());
	vm.reg(0) = result.unwrap();
	return Ok(Value{ result.unwrap() });
}

std::vector<uint8_t> MethodCall::serialize() const
{
	std::vector<uint8_t> result{
		METHOD_CALL,
		m_caller,
	};
	py::serialize(m_args, result);
	return result;
}