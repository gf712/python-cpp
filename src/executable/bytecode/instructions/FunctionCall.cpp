#include "FunctionCall.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"
#include "runtime/TypeError.hpp"

PyObject *execute(Interpreter &interpreter,
	PyObject *callable_object,
	PyTuple *args,
	PyDict *kwargs,
	PyDict *ns)
{
	std::string function_name;

	if (auto it = callable_object->attributes().find("__call__");
		it != callable_object->attributes().end()) {
		callable_object = it->second;
	}

	if (auto pyfunc = as<PyFunction>(callable_object)) {
		function_name = pyfunc->name();
	} else if (auto native_func = as<PyNativeFunction>(callable_object)) {
		function_name = native_func->name();
	} else if (auto method = as<PyBoundMethod>(callable_object)) {
		function_name = method->method()->name();
	} else if (auto method = as<PyMethodDescriptor>(callable_object)) {
		function_name = method->name()->value();
	} else {
		TODO();
	}

	if (auto pymethod = as<PyBoundMethod>(callable_object)) {
		// TODO: this is far from optimal, since we are creating a new vector
		//       just to append self to the start of the args tuple
		std::vector<Value> new_args_vector;
		new_args_vector.reserve(args->size() + 1);
		new_args_vector.push_back(pymethod->self());
		for (const auto &arg : args->elements()) { new_args_vector.push_back(arg); }
		args = PyTuple::create(new_args_vector);
		callable_object = pymethod->method();
	}

	if (auto pyfunc = as<PyFunction>(callable_object)) {

		auto function_locals = VirtualMachine::the().heap().allocate<PyDict>();
		auto *function_frame = ExecutionFrame::create(interpreter.execution_frame(),
			pyfunc->code()->register_count(),
			pyfunc->globals(),
			function_locals,
			ns);

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
					type_error(
						"{}() got an unexpected keyword argument '{}'", function_name, key_str.s);
					return nullptr;
				}
				auto &arg = function_frame->parameter(std::distance(argnames.begin(), arg_iter));
				if (arg.has_value()) {
					type_error(
						"{}() got multiple values for argument '{}'", function_name, key_str.s);
					return nullptr;
				}
				arg = value;
			}
		}

		spdlog::debug(
			"Requesting stack frame with {} virtual registers", pyfunc->code()->register_count());

		// spdlog::debug("Frame: {}", (void *)execution_frame);
		// spdlog::debug("Locals: {}", execution_frame->locals()->to_string());
		// spdlog::debug("Globals: {}", execution_frame->globals()->to_string());
		// if (ns) { spdlog::info("Namespace: {}", ns->to_string()); }
		return interpreter.call(pyfunc->code()->function(), function_frame);
	} else if (auto native_func = as<PyNativeFunction>(callable_object)) {
		return interpreter.call(native_func, args, kwargs);
	} else if (auto method = as<PyMethodDescriptor>(callable_object)) {
		std::vector<Value> new_args_vector;
		new_args_vector.reserve(args->size() - 1);
		PyObject *self = PyObject::from(args->elements()[0]);
		for (size_t i = 1; i < args->size(); ++i) {
			new_args_vector.push_back(args->elements()[i]);
		}
		args = PyTuple::create(new_args_vector);
		auto *result = method->method_descriptor()(self, args, kwargs);
		VirtualMachine::the().reg(0) = result;
		return result;
	} else {
		TODO();
	}
}

void FunctionCall::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto func = vm.reg(m_function_name);
	ASSERT(std::get_if<PyObject *>(&func));
	auto callable_object = std::get<PyObject *>(func);

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }

	auto *args_tuple = PyTuple::create(args);
	spdlog::debug("args_tuple: {}", (void *)&args_tuple);
	ASSERT(args_tuple);

	::execute(interpreter, callable_object, args_tuple, nullptr, nullptr);
}