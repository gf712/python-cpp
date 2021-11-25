#include "MethodCall.hpp"
#include "FunctionCall.hpp"

#include "runtime/AttributeError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyMethodWrapper.hpp"
#include "runtime/PySlotWrapper.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"


void MethodCall::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &maybe_self = vm.reg(m_caller);
	auto *obj = std::get<PyObject *>(maybe_self);
	ASSERT(obj)

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }

	auto *args_tuple = PyTuple::create(args);

	if (auto method_wrapper = as<PyMethodWrapper>(obj)) {
		// TODO: add support for static methods?
		ASSERT(args.size() > 0)
		PyObject *this_ = PyObject::from(args[0]);
		// TODO: add support for kwargs
		std::vector<Value> args_;
		for (size_t i = 1; i < args.size(); ++i) { args_.push_back(args[i]); }
		PyTuple *args = PyTuple::create(args_);
		PyDict *kwargs = nullptr;
		vm.reg(0) = method_wrapper->method_descriptor()(this_, args, kwargs);
	} else if (auto slot_wrapper = as<PySlotWrapper>(obj)) {
		// TODO: add support for static methods?
		ASSERT(args.size() > 0)
		PyObject *this_ = PyObject::from(args[0]);
		std::vector<Value> args_;
		for (size_t i = 1; i < args.size(); ++i) { args_.push_back(args[i]); }
		PyTuple *args = PyTuple::create(args_);
		PyDict *kwargs = nullptr;
		vm.reg(0) = slot_wrapper->slot()(this_, args, kwargs);
	} else {
		::execute(interpreter, obj, args_tuple, nullptr, nullptr);
	}
}