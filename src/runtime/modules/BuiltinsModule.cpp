#include "Modules.hpp"
#include "runtime/AssertionError.hpp"
#include "runtime/AttributeError.hpp"
#include "runtime/NameError.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyRange.hpp"
#include "runtime/PyStaticMethod.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"
#include "runtime/RuntimeError.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/ValueError.hpp"
#include "runtime/types/builtin.hpp"

#include "executable/Mangler.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/Bytecode.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "interpreter/Interpreter.hpp"
#include "memory/GarbageCollector.hpp"
#include "vm/VM.hpp"

#include "utilities.hpp"

#include <iostream>

using namespace py;

static PyModule *s_builtin_module = nullptr;

namespace {
// PyFunction *make_function(const std::string &function_name,
// 	int64_t function_id,
// 	const std::vector<std::string> &argnames,
// 	size_t argcount,
// 	PyModule *module,
// 	PyDict *globals)
// {
// 	auto &vm = VirtualMachine::the();
// 	auto function = std::static_pointer_cast<Bytecode>(vm.interpreter().function(function_name));
// 	PyCode *code = vm.heap().allocate<PyCode>(function,
// 		function_id,
// 		argnames,
// 		std::vector<Value>{},
// 		std::vector<Value>{},
// 		argcount,
// 		0,
// 		PyCode::CodeFlags::create(),
// 		module);
// 	return vm.heap().allocate<PyFunction>(function_name, code, globals);
// }


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
					type_error("sep must be None or a string, not {}", obj->type()->name()));
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
					type_error("end must be None or a string, not {}", obj->type()->name()));
				return nullptr;
			}
			end = std::get<String>(maybe_str).s;
		}
	}
	auto reprfunc = [](const auto &arg) { return arg->repr(); };

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
	if (kwargs) {
		interpreter.raise_exception(type_error("iter() takes no keyword arguments"));
		return py_none();
	}
	return arg->iter();
}


PyObject *next(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	ASSERT(args->size() == 1)
	if (kwargs) {
		interpreter.raise_exception(type_error("next() takes no keyword arguments"));
		return py_none();
	}
	const auto &arg = args->operator[](0);
	return arg->next();
}


PyObject *build_class(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	if (args->size() < 2) {
		interpreter.raise_exception(
			type_error("__build_class__: not enough arguments, got {}", args->size()));
		return nullptr;
	}
	// FIXME: should accept metaclass keyword
	ASSERT(!kwargs || kwargs->map().empty())
	auto *maybe_function_location = args->operator[](0);
	auto *mangled_class_name = args->operator[](1);
	spdlog::debug("__build_class__({}, {})",
		mangled_class_name->to_string(),
		maybe_function_location->to_string());

	if (!as<PyString>(mangled_class_name)) {
		interpreter.raise_exception(type_error("__build_class__: name is not a string"));
		return nullptr;
	}

	const auto mangled_class_name_as_string = as<PyString>(mangled_class_name)->value();

	PyFunction *callable = [&]() -> PyFunction * {
		if (as<PyInteger>(maybe_function_location)) {
			// auto function_id = std::get<int64_t>(pynumber->value().value);
			// FIXME: what should be the global dictionary for this?
			// FIXME: what should be the module for this?
			auto *f = interpreter.make_function(mangled_class_name_as_string, {}, {}, {});
			ASSERT(as<PyFunction>(f))
			return as<PyFunction>(f);
		} else if (auto *pyfunc = as<PyFunction>(maybe_function_location)) {
			return pyfunc;
		} else {
			interpreter.raise_exception(type_error("__build_class__: func must be callable"));
			return nullptr;
		}
	}();

	ASSERT(callable)

	if (!callable) { return nullptr; }

	std::vector<Value> bases_vector;
	if (args->size() > 2) {
		bases_vector.reserve(args->size() - 2);
		auto it = args->elements().begin() + 2;
		while (it != args->elements().end()) {
			bases_vector.push_back(*it);
			it++;
		}
	}

	auto *ns = PyDict::create();
	auto *bases = PyTuple::create(bases_vector);

	// this calls a function that defines a call
	// For example:
	// class A:
	// 	def foo(self):
	//		pass
	//
	// becomes something like this (in bytecode):
	//   1           0 LOAD_NAME                0 (__name__)
	//               2 STORE_NAME               1 (__module__)
	//               4 LOAD_CONST               0 ('A')
	//               6 STORE_NAME               2 (__qualname__)
	//
	//   2           8 LOAD_CONST               1 (<code object foo at 0x5557f27c0390, file
	//   "example.py", line 2>)
	//              10 LOAD_CONST               2 ('A.foo')
	//              12 MAKE_FUNCTION            0
	//              14 STORE_NAME               3 (foo)
	//              16 LOAD_CONST               3 (None)
	//              18 RETURN_VALUE
	// and calling these instructions creates the class' methods and attributes (i.e. foo)
	// call with frame keeps a reference to locals in a ns
	// so we have a reference to all class attributes and methods
	// i.e. {__module__: __name__, __qualname__: 'A', foo: <function A.foo>}
	callable->call_with_frame(ns, PyTuple::create(), PyDict::create());

	const std::string class_name_str =
		Mangler::default_mangler().class_demangle(mangled_class_name_as_string);

	auto *class_name = PyString::create(class_name_str);

	auto *call_args = PyTuple::create(class_name, bases, ns);

	// FIXME: determine what the actual metaclass is
	auto *metaclass = type();

	PyObject *cls = metaclass->__call__(call_args, nullptr);
	if (cls) { ASSERT(as<PyType>(cls)) }
	return cls;
}

PyObject *globals(const PyTuple *, const PyDict *, Interpreter &interpreter)
{
	return interpreter.execution_frame()->globals();
}


PyObject *locals(const PyTuple *, const PyDict *, Interpreter &interpreter)
{
	return interpreter.execution_frame()->locals();
}


PyObject *len(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	if (args->size() != 1) {
		interpreter.raise_exception(
			type_error("len() takes exactly one argument ({} given)", args->size()));
		return nullptr;
	}
	if (kwargs && !kwargs->map().empty()) {
		interpreter.raise_exception(type_error("len() takes no keyword arguments"));
		return nullptr;
	}
	return PyObject::from(args->elements()[0])->len();
}


PyObject *id(const PyTuple *args, const PyDict *, Interpreter &)
{
	ASSERT(args->size() == 1)
	return PyNumber::create(
		Number{ static_cast<int64_t>(bit_cast<intptr_t>(args->operator[](0))) });
}

PyObject *hasattr(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	if (args->size() != 2) {
		interpreter.raise_exception(
			type_error("hasattr expected 2 arguments, got {}", args->size()));
		return nullptr;
	}
	auto *obj = PyObject::from(args->elements()[0]);
	auto *name = PyObject::from(args->elements()[1]);
	if (!as<PyString>(name)) {
		interpreter.raise_exception(type_error("hasattr(): attribute name must be string"));
		return nullptr;
	}

	auto [_, found_status] = obj->lookup_attribute(name);
	if (found_status == LookupAttrResult::FOUND) {
		return py_true();
	} else if (found_status == LookupAttrResult::NOT_FOUND) {
		return py_false();
	} else {
		return nullptr;
	}
}

PyObject *getattr(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	if (args->size() != 2 && args->size() != 3) {
		interpreter.raise_exception(
			type_error("getattr expected 2 or 3 arguments, got {}", args->size()));
		return nullptr;
	}
	auto *obj = PyObject::from(args->elements()[0]);
	auto *name = PyObject::from(args->elements()[1]);
	if (!as<PyString>(name)) {
		interpreter.raise_exception(type_error("getattr(): attribute name must be string"));
		return nullptr;
	}

	if (args->size() == 2) {
		return obj->getattribute(name);
	} else {
		auto *default_value = PyObject::from(args->elements()[2]);
		auto [attr_value, found_status] = obj->lookup_attribute(name);
		if (found_status == LookupAttrResult::FOUND) {
			return attr_value;
		} else if (found_status == LookupAttrResult::NOT_FOUND) {
			return default_value;
		} else {
			return nullptr;
		}
	}
}

PyObject *hex(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	ASSERT(args->size() == 1)
	if (auto pynumber = PyNumber::as_number(args->operator[](0))) {
		if (std::holds_alternative<int64_t>(pynumber->value().value)) {
			return PyString::create(
				fmt::format("{0:#x}", std::get<int64_t>(pynumber->value().value)));
		} else {
			// FIXME: when float is separated from integer fix this
			interpreter.raise_exception(
				type_error("'float' object cannot be interpreted as an integer",
					args->operator[](0)->type()->name()));
		}
	} else {
		interpreter.raise_exception(type_error("'{}' object cannot be interpreted as an integer",
			args->operator[](0)->type()->name()));
	}
	return nullptr;
}

PyObject *ord(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	ASSERT(args->size() == 1)
	if (auto pystr = as<PyString>(args->operator[](0))) {
		if (auto codepoint = pystr->codepoint()) {
			return PyInteger::create(static_cast<int64_t>(*codepoint));
		} else {
			auto size = pystr->len();
			type_error("ord() expected a character, but string of length {} found",
				as<PyInteger>(size)->value().to_string());
		}
	} else {
		interpreter.raise_exception(type_error("ord() expected string of length 1, but {} found",
			args->operator[](0)->type()->name()));
	}
	return nullptr;
}

PyList *dir(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	ASSERT(args->size() < 2)
	auto dir_list = PyList::create();
	if (args->size() == 0) {
		for (const auto &[k, _] : interpreter.execution_frame()->locals()->map()) {
			dir_list->elements().push_back(PyObject::from(k));
		}
	} else {
		const auto &arg = args->elements()[0];

		// If the object is a module object, the list contains the names of the module’s attributes.
		if (std::holds_alternative<PyObject *>(arg) && as<PyModule>(std::get<PyObject *>(arg))) {
			auto *pymodule = as<PyModule>(std::get<PyObject *>(arg));
			for (const auto &[k, _] : pymodule->symbol_table()) {
				dir_list->elements().push_back(k);
			}
		}
		// If the object is a type or class object, the list contains the names of its attributes,
		// and recursively of the attributes of its bases.

		// Otherwise, the list contains the object’s attributes’ names, the names of its class’s
		// attributes, and recursively of the attributes of its class’s base classes.
		else {
			auto object = PyObject::from(arg);
			for (const auto &[k, _] : object->attributes().map()) {
				dir_list->elements().push_back(k);
			}
		}
	}

	dir_list->sort();
	return dir_list;
}

PyObject *repr(const PyTuple *args, const PyDict *, Interpreter &)
{
	if (args->size() != 1) {
		type_error("repr() takes exactly one argument ({} given)", args->size());
		return nullptr;
	}
	return PyObject::from(args->elements()[0])->__repr__();
}

PyObject *abs(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	if (args->size() != 1) {
		interpreter.raise_exception(
			type_error("abs() takes exactly one argument ({} given)", args->size()));
		return nullptr;
	}
	if (kwargs && !kwargs->map().empty()) {
		interpreter.raise_exception(type_error("abs() takes no keyword arguments"));
		return nullptr;
	}
	return PyObject::from(args->elements()[0])->abs();
}

PyObject *staticmethod(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	if (args->size() != 1) {
		interpreter.raise_exception(
			type_error("staticmethod() takes exactly one argument ({} given)", args->size()));
		return nullptr;
	}
	if (kwargs && !kwargs->map().empty()) {
		interpreter.raise_exception(type_error("staticmethod() takes no keyword arguments"));
		return nullptr;
	}

	auto *function = as<PyFunction>(PyObject::from(args->elements()[0]));

	return PyStaticMethod::create(PyString::create(function->function_name()), function);
}

PyObject *isinstance(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	if (args->size() != 2) {
		interpreter.raise_exception(
			type_error("isinstance expected 2 arguments, got {}", args->size()));
		return nullptr;
	}

	if (kwargs && !kwargs->map().empty()) {
		interpreter.raise_exception(type_error("isinstance() takes no keyword arguments"));
		return nullptr;
	}
	auto *object = PyObject::from(args->elements()[0]);
	auto *classinfo = PyObject::from(args->elements()[1]);

	if (auto *class_info_tuple = as<PyTuple>(classinfo)) {
		(void)class_info_tuple;
		TODO();
	} else if (auto *class_info_type = as<PyType>(classinfo)) {
		if (object->type() == class_info_type) {
			return py_true();
		} else {
			auto *mro = object->type()->mro();
			for (const auto &m : mro->elements()) {
				if (std::get<PyObject *>(m) == class_info_type) { return py_true(); }
			}
			return py_false();
		}
	} else {
		TODO();
	}
}

PyObject *issubclass(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	if (args->size() != 2) {
		interpreter.raise_exception(
			type_error("issubclass expected 2 arguments, got {}", args->size()));
		return nullptr;
	}

	if (kwargs && !kwargs->map().empty()) {
		interpreter.raise_exception(type_error("issubclass() takes no keyword arguments"));
		return nullptr;
	}
	auto *class_ = PyObject::from(args->elements()[0]);
	auto *classinfo = PyObject::from(args->elements()[1]);

	if (auto *class_info_tuple = as<PyTuple>(classinfo)) {
		(void)class_info_tuple;
		TODO();
	} else if (auto *class_info_type = as<PyType>(classinfo)) {
		auto *class_as_type = as<PyType>(class_);
		if (!class_as_type) {
			interpreter.raise_exception(type_error("issubclass() arg 1 must be a class"));
			return nullptr;
		}
		return class_as_type->issubclass(class_info_type) ? py_true() : py_false();
	} else {
		TODO();
	}
}

auto initialize_types()
{
	type();
	bool_();
	bytes();
	ellipsis();
	str();
	float_();
	integer();
	none();
	module();
	object();
	dict();
	dict_items();
	dict_items_iterator();
	list();
	list_iterator();
	tuple();
	tuple_iterator();
	range();
	range_iterator();
	function();
	native_function();
	code();
	cell();
	builtin_method();
	slot_wrapper();
	bound_method();
	method_wrapper();
	static_method();
	property();
	classmethod();

	return std::array{ type(),
		bool_(),
		bytes(),
		ellipsis(),
		str(),
		float_(),
		integer(),
		none(),
		object(),
		dict(),
		list(),
		tuple(),
		range(),
		property(),
		classmethod() };
}

auto initialize_exceptions(PyModule *blt)
{
	BaseException::register_type(blt);
	Exception::register_type(blt);
	TypeError::register_type(blt);
	AssertionError::register_type(blt);
	AttributeError::register_type(blt);
	StopIteration::register_type(blt);
	ValueError::register_type(blt);
	NameError::register_type(blt);
	RuntimeError::register_type(blt);
}

}// namespace

namespace py {

PyModule *builtins_module(Interpreter &interpreter)
{
	auto &heap = VirtualMachine::the().heap();

	// FIXME: second check (check address is valid) is only needed for unittests since each test
	// 		  clears the heap but is still the same executable (so it still uses the same static
	// 		  address)
	if (s_builtin_module && heap.slab().has_address(bit_cast<uint8_t *>(s_builtin_module))) {
		return s_builtin_module;
	}

	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();

	auto types = initialize_types();

	s_builtin_module = heap.allocate<PyModule>(PyString::create("__builtins__"));

	for (auto *type : types) { s_builtin_module->insert(PyString::create(type->name()), type); }

	initialize_exceptions(s_builtin_module);

	s_builtin_module->insert(PyString::create("__build_class__"),
		heap.allocate<PyNativeFunction>(
			"__build_class__", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return build_class(args, kwargs, interpreter);
			}));

	s_builtin_module->insert(PyString::create("abs"),
		heap.allocate<PyNativeFunction>("abs", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return abs(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("dir"),
		heap.allocate<PyNativeFunction>("dir", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return dir(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("getattr"),
		heap.allocate<PyNativeFunction>("getattr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return getattr(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("globals"),
		heap.allocate<PyNativeFunction>("globals", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return globals(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("hasattr"),
		heap.allocate<PyNativeFunction>("hasattr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return hasattr(args, kwargs, interpreter);
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

	s_builtin_module->insert(PyString::create("isinstance"),
		heap.allocate<PyNativeFunction>(
			"isinstance", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return isinstance(args, kwargs, interpreter);
			}));

	s_builtin_module->insert(PyString::create("issubclass"),
		heap.allocate<PyNativeFunction>(
			"issubclass", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return issubclass(args, kwargs, interpreter);
			}));

	s_builtin_module->insert(PyString::create("locals"),
		heap.allocate<PyNativeFunction>("locals", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return locals(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("len"),
		heap.allocate<PyNativeFunction>("len", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return len(args, kwargs, interpreter);
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

	s_builtin_module->insert(PyString::create("repr"),
		heap.allocate<PyNativeFunction>("repr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return repr(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("staticmethod"),
		heap.allocate<PyNativeFunction>(
			"staticmethod", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return staticmethod(args, kwargs, interpreter);
			}));

	return s_builtin_module;
}

}// namespace py