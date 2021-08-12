#include "CustomPyObject.hpp"
#include "PyBuiltins.hpp"
#include "PyDict.hpp"
#include "PyModule.hpp"
#include "PyNumber.hpp"
#include "PyRange.hpp"
#include "StopIterationException.hpp"

#include "bytecode/instructions/FunctionCall.hpp"
#include "bytecode/VM.hpp"
#include "interpreter/Interpreter.hpp"

#include <iostream>

namespace {
std::optional<int64_t> to_integer(const std::shared_ptr<PyObject> &obj, Interpreter &interpreter)
{
	if (auto pynumber = as<PyNumber>(obj)) {
		if (auto int_value = std::get_if<int64_t>(&pynumber->value().value)) { return *int_value; }
		interpreter.raise_exception(
			"TypeError: '{}' object cannot be interpreted as an integer", object_name(obj->type()));
	}
	return {};
}


std::shared_ptr<PyFunction> make_function(const std::string &function_name,
	int64_t function_id,
	const std::vector<std::string> &argnames)
{
	auto &vm = VirtualMachine::the();
	auto code = vm.heap().allocate<PyCode>(
		vm.function_offset(function_id), vm.function_register_count(function_id), argnames);
	return vm.heap().allocate<PyFunction>(function_name, std::static_pointer_cast<PyCode>(code));
}


std::shared_ptr<PyObject> print(const std::shared_ptr<PyTuple> &args,
	const std::shared_ptr<PyDict> &kwargs,
	Interpreter &interpreter)
{
	std::string separator = " ";
	if (kwargs) {
		static const Value separator_keyword = String{ "sep" };

		if (auto it = kwargs->map().find(separator_keyword); it != kwargs->map().end()) {
			auto maybe_str = it->second;
			if (!std::holds_alternative<String>(maybe_str)) {
				auto obj =
					std::visit([](const auto &value) { return PyObject::from(value); }, maybe_str);
				interpreter.raise_exception(
					"TypeError: sep must be None or a string, not {}", object_name(obj->type()));
				return nullptr;
			}
			separator = std::get<String>(maybe_str).s;
		}
	}
	auto reprfunc = [&interpreter](const auto &arg) {
		auto reprfunc = arg->slots().repr;
		if (std::holds_alternative<ReprSlotFunctionType>(reprfunc)) {
			auto repr_native = std::get<ReprSlotFunctionType>(reprfunc);
			spdlog::debug("Repr native function ptr: {}", static_cast<void *>(&repr_native));
			return repr_native();
		} else {
			auto pyfunc = std::get<std::shared_ptr<PyFunction>>(reprfunc);
			spdlog::debug("Repr native function ptr: {}", static_cast<void *>(pyfunc.get()));
			return execute(VirtualMachine::the(),
				interpreter,
				pyfunc,
				VirtualMachine::the().heap().allocate<PyTuple>(std::vector<Value>{ arg }),
				nullptr,
				nullptr);
		}
	};

	auto arg_it = args->begin();
	auto arg_it_end = args->end();
	if (arg_it == arg_it_end) {
		std::cout << std::endl;
		return py_none();
	}
	--arg_it_end;

	while (arg_it != arg_it_end) {
		spdlog::debug("arg function ptr: {}", static_cast<void *>((*arg_it).get()));
		auto reprobj = reprfunc(*arg_it);
		spdlog::debug("repr result: {}", reprobj->to_string());
		std::cout << reprobj->to_string() << separator;
		std::advance(arg_it, 1);
	}

	spdlog::debug("arg function ptr: {}", static_cast<void *>((*arg_it).get()));
	auto reprobj = reprfunc(*arg_it);
	spdlog::debug("repr result: {}", reprobj->to_string());
	std::cout << reprobj->to_string();

	// make sure this is flushed immediately with a newline
	std::cout << std::endl;
	return py_none();
}


std::shared_ptr<PyObject> iter(const std::shared_ptr<PyTuple> &args,
	const std::shared_ptr<PyDict> &kwargs,
	Interpreter &interpreter)
{
	ASSERT(args->size() == 1)
	const auto &arg = args->operator[](0);
	auto iterfunc = arg->slots().iter;
	if (kwargs) {
		interpreter.raise_exception("TypeError: iter() takes no keyword arguments");
		return py_none();
	}
	if (std::holds_alternative<ReprSlotFunctionType>(iterfunc)) {
		auto iter_native = std::get<ReprSlotFunctionType>(iterfunc);
		spdlog::debug("Iter native function ptr: {}", static_cast<void *>(&iter_native));
		return iter_native();
	} else {
		auto pyfunc = std::get<std::shared_ptr<PyFunction>>(iterfunc);
		spdlog::debug("Iter native function ptr: {}", static_cast<void *>(pyfunc.get()));
		return execute(VirtualMachine::the(),
			interpreter,
			pyfunc,
			VirtualMachine::the().heap().allocate<PyTuple>(std::vector<Value>{ arg }),
			nullptr,
			nullptr);
	}
}


std::shared_ptr<PyObject> next(const std::shared_ptr<PyTuple> &args,
	const std::shared_ptr<PyDict> &kwargs,
	Interpreter &interpreter)
{
	ASSERT(args->size() == 1)
	if (kwargs) {
		interpreter.raise_exception("TypeError: next() takes no keyword arguments");
		return py_none();
	}
	std::shared_ptr<PyObject> next_result{ nullptr };
	const auto &arg = args->operator[](0);
	auto iterfunc = arg->slots().iter;
	if (std::holds_alternative<ReprSlotFunctionType>(iterfunc)) {
		auto iter_native = std::get<ReprSlotFunctionType>(iterfunc);
		spdlog::debug("Iter native function ptr: {}", static_cast<void *>(&iter_native));
		next_result = iter_native();
	} else {
		auto pyfunc = std::get<std::shared_ptr<PyFunction>>(iterfunc);
		spdlog::debug("Iter native function ptr: {}", static_cast<void *>(pyfunc.get()));
		next_result = execute(VirtualMachine::the(),
			interpreter,
			pyfunc,
			VirtualMachine::the().heap().allocate<PyTuple>(std::vector<Value>{ arg }),
			nullptr,
			nullptr);
	}

	if (!next_result) { interpreter.raise_exception(stop_iteration("")); }
	return next_result;
}


std::shared_ptr<PyObject> range(const std::shared_ptr<PyTuple> &args,
	const std::shared_ptr<PyDict> &,
	Interpreter &interpreter)
{
	ASSERT(args->size() == 1)
	const auto &arg = args->operator[](0);
	auto &heap = VirtualMachine::the().heap();
	if (auto pynumber = to_integer(arg, interpreter)) {
		return std::static_pointer_cast<PyObject>(heap.allocate<PyRange>(*pynumber));
	}
	return py_none();
}


std::shared_ptr<PyObject> build_class(const std::shared_ptr<PyTuple> &args,
	const std::shared_ptr<PyDict> &,
	Interpreter &interpreter)
{
	ASSERT(args->size() == 2)
	const auto &class_name = args->operator[](0);
	const auto &function_location = args->operator[](1);
	spdlog::debug(
		"__build_class__({}, {})", class_name->to_string(), function_location->to_string());

	ASSERT(as<PyString>(class_name))
	auto class_name_as_string = as<PyString>(class_name)->value();

	ASSERT(as<PyNumber>(function_location))
	auto pynumber = as<PyNumber>(function_location)->value();
	ASSERT(std::get_if<int64_t>(&pynumber.value))
	auto function_id = std::get<int64_t>(pynumber.value);

	auto pyfunc = make_function(class_name_as_string, function_id, std::vector<std::string>{});

	auto &vm = VirtualMachine::the();

	return vm.heap().allocate<PyNativeFunction>(class_name_as_string,
		[class_name, &interpreter, pyfunc, class_name_as_string](
			const std::shared_ptr<PyTuple> &call_args, const std::shared_ptr<PyDict> &call_kwargs) {
			spdlog::debug("Calling __build_class__");

			std::vector args_vector{ class_name };
			for (const auto &arg : *call_args) { args_vector.push_back(arg); }

			auto &vm = VirtualMachine::the();
			auto class_args = vm.heap().allocate<PyTuple>(args_vector);

			auto ns = vm.heap().allocate<PyDict>();
			execute(vm, interpreter, pyfunc, class_args, call_kwargs, ns);

			CustomPyObjectContext ctx{ class_name_as_string, ns };
			return vm.heap().allocate<CustomPyObject>(ctx, std::shared_ptr<PyTuple>{});
		});
}

std::shared_ptr<PyObject> globals(const std::shared_ptr<PyTuple> &,
	const std::shared_ptr<PyDict> &,
	Interpreter &interpreter)
{
	return interpreter.execution_frame()->globals();
}


std::shared_ptr<PyObject> locals(const std::shared_ptr<PyTuple> &,
	const std::shared_ptr<PyDict> &,
	Interpreter &interpreter)
{
	return interpreter.execution_frame()->locals();
}
}// namespace


std::shared_ptr<PyModule> fetch_builtins(Interpreter &interpreter)
{
	auto &heap = VirtualMachine::the().heap();
	auto builtin_module = heap.allocate<PyModule>(PyString::create("builtins"));

	builtin_module->insert(PyString::create("print"),
		heap.allocate<PyNativeFunction>("print",
			[&interpreter](
				const std::shared_ptr<PyTuple> &args, const std::shared_ptr<PyDict> &kwargs) {
				return print(args, kwargs, interpreter);
			}));

	builtin_module->insert(PyString::create("iter"),
		heap.allocate<PyNativeFunction>("iter",
			[&interpreter](
				const std::shared_ptr<PyTuple> &args, const std::shared_ptr<PyDict> &kwargs) {
				return iter(args, kwargs, interpreter);
			}));

	builtin_module->insert(PyString::create("next"),
		heap.allocate<PyNativeFunction>("next",
			[&interpreter](
				const std::shared_ptr<PyTuple> &args, const std::shared_ptr<PyDict> &kwargs) {
				return next(args, kwargs, interpreter);
			}));

	builtin_module->insert(PyString::create("range"),
		heap.allocate<PyNativeFunction>("range",
			[&interpreter](
				const std::shared_ptr<PyTuple> &args, const std::shared_ptr<PyDict> &kwargs) {
				return range(args, kwargs, interpreter);
			}));

	builtin_module->insert(PyString::create("__build_class__"),
		heap.allocate<PyNativeFunction>("__build_class__",
			[&interpreter](
				const std::shared_ptr<PyTuple> &args, const std::shared_ptr<PyDict> &kwargs) {
				return build_class(args, kwargs, interpreter);
			}));

	builtin_module->insert(PyString::create("globals"),
		heap.allocate<PyNativeFunction>("globals",
			[&interpreter](
				const std::shared_ptr<PyTuple> &args, const std::shared_ptr<PyDict> &kwargs) {
				return globals(args, kwargs, interpreter);
			}));

	builtin_module->insert(PyString::create("locals"),
		heap.allocate<PyNativeFunction>("locals",
			[&interpreter](
				const std::shared_ptr<PyTuple> &args, const std::shared_ptr<PyDict> &kwargs) {
				return locals(args, kwargs, interpreter);
			}));

	return builtin_module;
}