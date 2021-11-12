#include "Modules.hpp"
#include "runtime/CustomPyObject.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyRange.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/StopIterationException.hpp"
#include "runtime/TypeError.hpp"

#include "executable/bytecode/Bytecode.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

#include "utilities.hpp"

#include <iostream>

static PyModule *s_builtin_module = nullptr;

namespace {
std::optional<int64_t> to_integer(const PyObject *obj, Interpreter &interpreter)
{
	if (auto pynumber = as<PyNumber>(obj)) {
		if (auto int_value = std::get_if<int64_t>(&pynumber->value().value)) { return *int_value; }
		interpreter.raise_exception(
			"TypeError: '{}' object cannot be interpreted as an integer", object_name(obj->type()));
	}
	return {};
}


PyFunction *make_function(const std::string &function_name,
	int64_t function_id,
	const std::vector<std::string> &argnames,
	PyModule *module,
	PyDict *globals)
{
	auto &vm = VirtualMachine::the();
	ASSERT(vm.interpreter().functions(function_id)->backend() == FunctionExecutionBackend::BYTECODE)
	auto function = std::static_pointer_cast<Bytecode>(vm.interpreter().functions(function_id));
	PyCode *code = vm.heap().allocate<PyCode>(function, function_id, argnames, module);
	return vm.heap().allocate<PyFunction>(function_name, code, globals);
}


PyObject *print(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	std::string separator = " ";
	std::string end = "\n";
	if (kwargs) {
		static const Value separator_keyword = String{ "sep" };
		static const Value end_keyword = String{ "end" };

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
		if (auto it = kwargs->map().find(end_keyword); it != kwargs->map().end()) {
			auto maybe_str = it->second;
			if (!std::holds_alternative<String>(maybe_str)) {
				auto obj =
					std::visit([](const auto &value) { return PyObject::from(value); }, maybe_str);
				interpreter.raise_exception(
					"TypeError: end must be None or a string, not {}", object_name(obj->type()));
				return nullptr;
			}
			end = std::get<String>(maybe_str).s;
		}
	}
	auto reprfunc = [&interpreter](const auto &arg) {
		auto reprfunc = arg->slots().repr;
		if (std::holds_alternative<ReprSlotFunctionType>(reprfunc)) {
			auto repr_native = std::get<ReprSlotFunctionType>(reprfunc);
			spdlog::debug("Repr native function ptr: {}", static_cast<void *>(&repr_native));
			return repr_native();
		} else {
			auto pyfunc = std::get<PyFunction *>(reprfunc);
			spdlog::debug("Repr native function ptr: {}", static_cast<void *>(pyfunc));
			return execute(interpreter,
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
		spdlog::debug("arg function ptr: {}", static_cast<void *>(*arg_it));
		auto reprobj = reprfunc(*arg_it);
		spdlog::debug("repr result: {}", reprobj->to_string());
		std::cout << reprobj->to_string() << separator;
		std::advance(arg_it, 1);
	}

	spdlog::debug("arg function ptr: {}", static_cast<void *>(*arg_it));
	auto reprobj = reprfunc(*arg_it);
	spdlog::debug("repr result: {}", reprobj->to_string());
	std::cout << reprobj->to_string() << end;

	return py_none();
}


PyObject *iter(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	ASSERT(args->size() == 1)
	const auto &arg = args->operator[](0);
	auto iterfunc = arg->slots().iter;
	if (kwargs) {
		interpreter.raise_exception("TypeError: iter() takes no keyword arguments");
		return py_none();
	}
	if (std::holds_alternative<IterSlotFunctionType>(iterfunc)) {
		auto iter_native = std::get<IterSlotFunctionType>(iterfunc);
		spdlog::debug("Iter native function ptr: {}", static_cast<void *>(&iter_native));
		return iter_native();
	} else {
		auto pyfunc = std::get<PyFunction *>(iterfunc);
		spdlog::debug("Iter native function ptr: {}", static_cast<void *>(pyfunc));
		return execute(interpreter,
			pyfunc,
			VirtualMachine::the().heap().allocate<PyTuple>(std::vector<Value>{ arg }),
			nullptr,
			nullptr);
	}
}


PyObject *next(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	ASSERT(args->size() == 1)
	if (kwargs) {
		interpreter.raise_exception("TypeError: next() takes no keyword arguments");
		return py_none();
	}
	PyObject *next_result{ nullptr };
	const auto &arg = args->operator[](0);
	auto iterfunc = arg->slots().iter;
	if (std::holds_alternative<ReprSlotFunctionType>(iterfunc)) {
		auto iter_native = std::get<ReprSlotFunctionType>(iterfunc);
		spdlog::debug("Iter native function ptr: {}", static_cast<void *>(&iter_native));
		next_result = iter_native();
	} else {
		auto pyfunc = std::get<PyFunction *>(iterfunc);
		spdlog::debug("Iter native function ptr: {}", static_cast<void *>(pyfunc));
		next_result = execute(interpreter,
			pyfunc,
			VirtualMachine::the().heap().allocate<PyTuple>(std::vector<Value>{ arg }),
			nullptr,
			nullptr);
	}

	if (!next_result) { interpreter.raise_exception(stop_iteration("")); }
	return next_result;
}


PyObject *range(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	ASSERT(args->size() == 1)
	const auto &arg = args->operator[](0);
	auto &heap = VirtualMachine::the().heap();
	if (auto pynumber = to_integer(arg, interpreter)) { return heap.allocate<PyRange>(*pynumber); }
	return py_none();
}


PyObject *build_class(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	ASSERT(args->size() == 2)
	auto *class_name = args->operator[](0);
	auto *function_location = args->operator[](1);
	spdlog::debug(
		"__build_class__({}, {})", class_name->to_string(), function_location->to_string());

	ASSERT(as<PyString>(class_name))
	auto class_name_as_string = as<PyString>(class_name)->value();

	ASSERT(as<PyNumber>(function_location))
	auto pynumber = as<PyNumber>(function_location)->value();
	ASSERT(std::get_if<int64_t>(&pynumber.value))
	auto function_id = std::get<int64_t>(pynumber.value);

	// FIXME: what should be the global dictionary for this?
	// FIXME: what should be the module for this?
	auto *pyfunc = make_function(class_name_as_string,
		function_id,
		std::vector<std::string>{},
		nullptr,
		interpreter.execution_frame()->globals());

	auto &vm = VirtualMachine::the();

	// TODO: fix tracking of lambda captures
	return vm.heap().allocate<PyNativeFunction>(
		class_name_as_string,
		[class_name, &interpreter, pyfunc, class_name_as_string](
			const PyTuple *call_args, PyDict *call_kwargs) {
			spdlog::debug("Calling __build_class__");

			std::vector args_vector{ class_name };
			for (const auto &arg : *call_args) { args_vector.push_back(arg); }

			auto &vm = VirtualMachine::the();
			auto class_args = vm.heap().allocate<PyTuple>(args_vector);

			auto *ns = vm.heap().allocate<PyDict>();
			execute(interpreter, pyfunc, class_args, call_kwargs, ns);

			CustomPyObjectContext ctx{ class_name_as_string, ns };
			return vm.heap().allocate<CustomPyObject>(ctx, PyTuple::create());
		},
		class_name,
		pyfunc);
}

PyObject *globals(const PyTuple *, const PyDict *, Interpreter &interpreter)
{
	return interpreter.execution_frame()->globals();
}


PyObject *locals(const PyTuple *, const PyDict *, Interpreter &interpreter)
{
	return interpreter.execution_frame()->locals();
}


PyObject *id(const PyTuple *args, const PyDict *, Interpreter &)
{
	ASSERT(args->size() == 1)
	return PyNumber::create(
		Number{ static_cast<int64_t>(bit_cast<intptr_t>(args->operator[](0))) });
}

PyObject *hex(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	ASSERT(args->size() == 1)
	if (auto pynumber = as<PyNumber>(args->operator[](0))) {
		if (std::holds_alternative<int64_t>(pynumber->value().value)) {
			return PyString::create(
				fmt::format("{0:#x}", std::get<int64_t>(pynumber->value().value)));
		} else {
			// FIXME: when float is separated from integer fix this
			interpreter.raise_exception(
				"TypeError: 'float' object cannot be interpreted as an integer",
				object_name(args->operator[](0)->type()));
		}
	} else {
		interpreter.raise_exception("TypeError: '{}' object cannot be interpreted as an integer",
			object_name(args->operator[](0)->type()));
	}
	return nullptr;
}

PyObject *ord(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	ASSERT(args->size() == 1)
	if (auto pystr = as<PyString>(args->operator[](0))) {
		if (auto codepoint = pystr->codepoint()) {
			return PyNumber::create(Number{ static_cast<int64_t>(*codepoint) });
		} else {
			auto size = pystr->len_impl(interpreter);
			type_error("ord() expected a character, but string of length {} found",
				as<PyNumber>(size)->value().to_string());
		}
	} else {
		interpreter.raise_exception("TypeError: ord() expected string of length 1, but {} found",
			object_name(args->operator[](0)->type()));
	}
	return nullptr;
}

PyList *dir(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	ASSERT(args->size() < 2)
	auto dir_list = PyList::create();
	if (args->size() == 0) {
		for (const auto &[k, _] : interpreter.execution_frame()->locals()->map()) {
			dir_list->append(PyObject::from(k));
		}
	} else {
		const auto &arg = args->elements()[0];

		// If the object is a module object, the list contains the names of the module’s attributes.
		if (std::holds_alternative<PyObject *>(arg) && as<PyModule>(std::get<PyObject *>(arg))) {
			auto *pymodule = as<PyModule>(std::get<PyObject *>(arg));
			for (const auto &[k, _] : pymodule->symbol_table()) {
				dir_list->append(k); 
			}
		}
		// If the object is a type or class object, the list contains the names of its attributes,
		// and recursively of the attributes of its bases.

		// Otherwise, the list contains the object’s attributes’ names, the names of its class’s
		// attributes, and recursively of the attributes of its class’s base classes.
		else {
			auto object = PyObject::from(arg);
			for (const auto &[k, _] : object->attributes()) {
				dir_list->append(PyString::create(k));
			}
		}
	}
	// FIXME, this list should be ordered alphabetically
	return dir_list;
}
}// namespace


PyModule *builtins_module(Interpreter &interpreter)
{
	auto &heap = VirtualMachine::the().heap();

	if (s_builtin_module && heap.slab().has_address(bit_cast<uint8_t *>(s_builtin_module))) {
		return s_builtin_module;
	}

	s_builtin_module = heap.allocate<PyModule>(PyString::create("__builtins__"));

	s_builtin_module->insert(PyString::create("__build_class__"),
		heap.allocate<PyNativeFunction>(
			"__build_class__", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return build_class(args, kwargs, interpreter);
			}));

	s_builtin_module->insert(PyString::create("dir"),
		heap.allocate<PyNativeFunction>("dir", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return dir(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("globals"),
		heap.allocate<PyNativeFunction>("globals", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return globals(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("hex"),
		heap.allocate<PyNativeFunction>("hex", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return hex(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("id"),
		heap.allocate<PyNativeFunction>("id", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return id(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("iter"),
		heap.allocate<PyNativeFunction>("iter", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return iter(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("locals"),
		heap.allocate<PyNativeFunction>("locals", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return locals(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("next"),
		heap.allocate<PyNativeFunction>("next", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return next(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("ord"),
		heap.allocate<PyNativeFunction>("ord", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return ord(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("print"),
		heap.allocate<PyNativeFunction>("print", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return print(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("range"),
		heap.allocate<PyNativeFunction>("range", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return range(args, kwargs, interpreter);
		}));

	return s_builtin_module;
}