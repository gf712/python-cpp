#include "MethodCall.hpp"
#include "FunctionCall.hpp"

#include "runtime/AttributeError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"


void MethodCall::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	// auto func = vm.reg(m_function_name);
	// ASSERT(std::get_if<PyObject *>(&func));
	// auto *function_object = std::get<PyObject *>(func);

	// std::vector<Value> args;
	// // self
	// auto maybe_self = interpreter.execution_frame()->locals()->map().at(
	// 	PyString::from(String{ m_instance_name }));

	// ASSERT(std::holds_alternative<PyObject *>(maybe_self))
	// auto *obj = std::get<PyObject *>(maybe_self);
	// ASSERT(obj)

	// if (obj->type() == PyObjectType::PY_MODULE) {
	// 	// FIXME: does this need anything?
	// } else {
	// 	auto *self = obj;
	// 	args.push_back(self);
	// }
	// for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }

	// auto *args_tuple = PyTuple::create(args);

	// ASSERT(args_tuple);

	// // TODO: add support for kwargs
	// ::execute(interpreter, function_object, args_tuple, nullptr, nullptr);

	const auto &maybe_self = vm.reg(m_caller);
	auto *obj = std::get<PyObject *>(maybe_self);
	ASSERT(obj)

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }

	auto *args_tuple = PyTuple::create(args);

	if (auto method_wrapper = as<PyMethodDescriptor>(obj)) {
		// TODO: add support for static methods?
		ASSERT(args.size() > 0)
		PyObject *this_ = PyObject::from(args[0]);
		// TODO: add support for args and kwargs
		PyTuple *args = nullptr;
		PyDict *kwargs = nullptr;
		vm.reg(0) = method_wrapper->method_descriptor()(this_, args, kwargs);
	} else if (auto slot_wrapper = as<PySlotWrapper>(obj)) {
		// TODO: add support for static methods?
		ASSERT(args.size() > 0)
		PyObject *this_ = PyObject::from(args[0]);
		// TODO: add support for args and kwargs
		PyTuple *args = nullptr;
		PyDict *kwargs = nullptr;
		vm.reg(0) = slot_wrapper->slot()(this_, args, kwargs);
	} else {
		::execute(interpreter, obj, args_tuple, nullptr, nullptr);
	}
}