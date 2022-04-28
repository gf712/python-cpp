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


PyResult print(const PyTuple *args, const PyDict *kwargs, Interpreter &)
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
				if (obj.is_err()) return obj;
				return PyResult::Err(type_error("sep must be None or a string, not {}",
					obj.unwrap_as<PyObject>()->type()->name()));
			}
			separator = std::get<String>(maybe_str).s;
		}
		if (auto it = kwargs->map().find(end_keyword); it != kwargs->map().end()) {
			auto maybe_str = it->second;
			if (!std::holds_alternative<String>(maybe_str)) {
				auto obj =
					std::visit([](const auto &value) { return PyObject::from(value); }, maybe_str);
				if (obj.is_err()) return obj;
				return PyResult::Err(type_error("end must be None or a string, not {}",
					obj.unwrap_as<PyObject>()->type()->name()));
			}
			end = std::get<String>(maybe_str).s;
		}
	}
	auto reprfunc = [](const PyResult &arg) {
		if (arg.is_err()) return arg;
		return arg.unwrap_as<PyObject>()->repr();
	};

	auto arg_it = args->begin();
	auto arg_it_end = args->end();
	if (arg_it == arg_it_end) {
		std::cout << std::endl;
		return PyResult::Ok(py_none());
	}
	--arg_it_end;

	while (arg_it != arg_it_end) {
		spdlog::debug("arg function ptr: {}", static_cast<void *>((*arg_it).unwrap_as<PyObject>()));
		auto reprobj_ = reprfunc(*arg_it);
		if (reprobj_.is_err()) { return reprobj_; }
		auto reprobj = reprobj_.unwrap_as<PyString>();
		spdlog::debug("repr result: {}", reprobj->to_string());
		std::cout << reprobj->to_string() << separator;
		std::advance(arg_it, 1);
	}

	spdlog::debug("arg function ptr: {}", static_cast<void *>((*arg_it).unwrap_as<PyObject>()));
	auto reprobj_ = reprfunc(*arg_it);
	if (reprobj_.is_err()) { return reprobj_; }
	auto reprobj = reprobj_.unwrap_as<PyString>();
	spdlog::debug("repr result: {}", reprobj->to_string());
	std::cout << reprobj->to_string() << end;

	return PyResult::Ok(py_none());
}


PyResult iter(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	ASSERT(args->size() == 1)
	const auto &arg = args->operator[](0);
	if (kwargs) { return PyResult::Err(type_error("iter() takes no keyword arguments")); }
	return arg.and_then<PyObject>([](auto *obj) { return obj->iter(); });
}


PyResult next(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	ASSERT(args->size() == 1)
	if (kwargs) { return PyResult::Err(type_error("next() takes no keyword arguments")); }
	const auto &arg = args->operator[](0);
	return arg.and_then<PyObject>([](auto *obj) { return obj->next(); });
}


PyResult build_class(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	if (args->size() < 2) {
		return PyResult::Err(
			type_error("__build_class__: not enough arguments, got {}", args->size()));
	}
	// FIXME: should accept metaclass keyword
	ASSERT(!kwargs || kwargs->map().empty())
	auto maybe_function_location_ = args->operator[](0);
	if (maybe_function_location_.is_err()) return maybe_function_location_;
	auto *maybe_function_location = maybe_function_location_.unwrap_as<PyObject>();
	auto mangled_class_name_ = args->operator[](1);
	if (mangled_class_name_.is_err()) return mangled_class_name_;
	auto *mangled_class_name = mangled_class_name_.unwrap_as<PyObject>();
	spdlog::debug("__build_class__({}, {})",
		mangled_class_name->to_string(),
		maybe_function_location->to_string());

	if (!as<PyString>(mangled_class_name)) {
		return PyResult::Err(type_error("__build_class__: name is not a string"));
	}

	const auto mangled_class_name_as_string = as<PyString>(mangled_class_name)->value();

	PyResult callable = [&]() -> PyResult {
		if (as<PyInteger>(maybe_function_location)) {
			// auto function_id = std::get<int64_t>(pynumber->value().value);
			// FIXME: what should be the global dictionary for this?
			// FIXME: what should be the module for this?
			auto *f = interpreter.make_function(mangled_class_name_as_string, {}, {}, {});
			ASSERT(as<PyFunction>(f))
			return PyResult::Ok(as<PyFunction>(f));
		} else if (auto *pyfunc = as<PyFunction>(maybe_function_location)) {
			return PyResult::Ok(pyfunc);
		} else {
			return PyResult::Err(type_error("__build_class__: func must be callable"));
		}
	}();

	ASSERT(callable.is_ok())

	if (callable.is_err()) { TODO(); }

	std::vector<Value> bases_vector;
	if (args->size() > 2) {
		bases_vector.reserve(args->size() - 2);
		auto it = args->elements().begin() + 2;
		while (it != args->elements().end()) {
			bases_vector.push_back(*it);
			it++;
		}
	}

	auto ns_ = PyDict::create();
	if (ns_.is_err()) { return ns_; }
	auto *ns = ns_.unwrap_as<PyDict>();

	auto bases_ = PyTuple::create(bases_vector);
	if (bases_.is_err()) { return bases_; }
	auto *bases = bases_.unwrap_as<PyTuple>();

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
	auto args_ = PyTuple::create();
	if (args_.is_err()) { return args_; }
	auto *empty_args = args_.unwrap_as<PyTuple>();

	auto kwargs_ = PyDict::create();
	if (kwargs_.is_err()) { return kwargs_; }
	auto *empty_kwargs = kwargs_.unwrap_as<PyDict>();
	callable.unwrap_as<PyFunction>()->call_with_frame(ns, empty_args, empty_kwargs);

	const std::string class_name_str =
		Mangler::default_mangler().class_demangle(mangled_class_name_as_string);

	auto class_name = PyString::create(class_name_str);
	if (class_name.is_err()) { return class_name; }

	auto call_args = PyTuple::create(class_name.unwrap_as<PyString>(), bases, ns);
	if (call_args.is_err()) { return call_args; }

	// FIXME: determine what the actual metaclass is
	auto *metaclass = type();

	auto cls = metaclass->__call__(call_args.unwrap_as<PyTuple>(), nullptr);
	if (cls.is_ok()) { ASSERT(cls.unwrap_as<PyType>()) }
	return cls;
}

PyResult globals(const PyTuple *, const PyDict *, Interpreter &interpreter)
{
	return PyResult::Ok(interpreter.execution_frame()->globals());
}


PyResult locals(const PyTuple *, const PyDict *, Interpreter &interpreter)
{
	return PyResult::Ok(interpreter.execution_frame()->locals());
}


PyResult len(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (args->size() != 1) {
		return PyResult::Err(
			type_error("len() takes exactly one argument ({} given)", args->size()));
	}
	if (kwargs && !kwargs->map().empty()) {
		return PyResult::Err(type_error("len() takes no keyword arguments"));
	}

	return PyObject::from(args->elements()[0]).and_then<PyObject>([](PyObject *o) {
		return o->len();
	});
}

PyResult id(const PyTuple *args, const PyDict *, Interpreter &)
{
	ASSERT(args->size() == 1)
	auto obj = args->operator[](0);
	if (obj.is_err()) return obj;
	return PyNumber::create(
		Number{ static_cast<int64_t>(bit_cast<intptr_t>(obj.unwrap_as<PyObject>())) });
}

PyResult hasattr(const PyTuple *args, const PyDict *, Interpreter &)
{
	if (args->size() != 2) {
		return PyResult::Err(type_error("hasattr expected 2 arguments, got {}", args->size()));
	}
	auto obj_ = PyObject::from(args->elements()[0]);
	if (obj_.is_err()) return obj_;
	auto *obj = obj_.unwrap_as<PyObject>();
	auto name_ = PyObject::from(args->elements()[1]);
	if (name_.is_err()) return name_;
	auto *name = name_.unwrap_as<PyObject>();
	if (!as<PyString>(name)) {
		return PyResult::Err(type_error("hasattr(): attribute name must be string"));
	}

	auto [result, found_status] = obj->lookup_attribute(name);
	if (found_status == LookupAttrResult::FOUND) {
		return PyResult::Ok(py_true());
	} else if (found_status == LookupAttrResult::NOT_FOUND) {
		return PyResult::Ok(py_false());
	} else {
		return result;
	}
}

PyResult getattr(const PyTuple *args, const PyDict *, Interpreter &)
{
	if (args->size() != 2 && args->size() != 3) {
		return PyResult::Err(type_error("getattr expected 2 or 3 arguments, got {}", args->size()));
	}
	auto obj_ = PyObject::from(args->elements()[0]);
	if (obj_.is_err()) return obj_;
	auto *obj = obj_.unwrap_as<PyObject>();
	auto name_ = PyObject::from(args->elements()[1]);
	if (name_.is_err()) return name_;
	auto *name = name_.unwrap_as<PyObject>();
	if (!as<PyString>(name)) {
		return PyResult::Err(type_error("getattr(): attribute name must be string"));
	}

	if (args->size() == 2) {
		return obj->getattribute(name);
	} else {
		auto default_value_ = PyObject::from(args->elements()[2]);
		if (default_value_.is_err()) return default_value_;
		auto *default_value = default_value_.unwrap_as<PyObject>();

		auto [attr_value, found_status] = obj->lookup_attribute(name);

		if (attr_value.is_err()) { return attr_value; }

		if (found_status == LookupAttrResult::FOUND) {
			return attr_value;
		} else {
			return PyResult::Ok(default_value);
		}
	}
}

PyResult setattr(const PyTuple *args, const PyDict *, Interpreter &)
{
	if (args->size() != 3) {
		return PyResult::Err(type_error("setattr expected 3 arguments, got {}", args->size()));
	}
	auto obj_ = PyObject::from(args->elements()[0]);
	if (obj_.is_err()) return obj_;
	auto *obj = obj_.unwrap_as<PyObject>();
	auto name_ = PyObject::from(args->elements()[1]);
	if (name_.is_err()) return name_;
	auto *name = name_.unwrap_as<PyObject>();
	auto value_ = PyObject::from(args->elements()[1]);
	if (value_.is_err()) return value_;
	auto *value = value_.unwrap_as<PyObject>();

	if (!as<PyString>(name)) {
		return PyResult::Err(type_error("setattr(): attribute name must be string"));
	}

	return obj->setattribute(name, value).and_then<PyObject>([](PyObject *) {
		return PyResult::Ok(py_none());
	});
}

PyResult hex(const PyTuple *args, const PyDict *, Interpreter &)
{
	ASSERT(args->size() == 1)
	auto obj_ = args->operator[](0);
	if (obj_.is_err()) return obj_;
	auto *obj = obj_.unwrap_as<PyObject>();
	if (auto pynumber = PyNumber::as_number(obj)) {
		if (std::holds_alternative<int64_t>(pynumber->value().value)) {
			return PyString::create(
				fmt::format("{0:#x}", std::get<int64_t>(pynumber->value().value)));
		} else {
			// FIXME: when float is separated from integer fix this
			return PyResult::Err(type_error(
				"'float' object cannot be interpreted as an integer", obj->type()->name()));
		}
	} else {
		return PyResult::Err(
			type_error("'{}' object cannot be interpreted as an integer", obj->type()->name()));
	}
}

PyResult ord(const PyTuple *args, const PyDict *, Interpreter &)
{
	ASSERT(args->size() == 1)
	auto obj_ = args->operator[](0);
	if (obj_.is_err()) return obj_;
	auto *obj = obj_.unwrap_as<PyObject>();
	if (auto pystr = as<PyString>(obj)) {
		if (auto codepoint = pystr->codepoint()) {
			return PyInteger::create(static_cast<int64_t>(*codepoint));
		} else {
			auto size = pystr->len();
			if (size.is_err()) { return size; }
			return PyResult::Err(
				type_error("ord() expected a character, but string of length {} found",
					size.unwrap_as<PyObject>()->to_string()));
		}
	} else {
		return PyResult::Err(
			type_error("ord() expected string of length 1, but {} found", obj->type()->name()));
	}
}

PyResult dir(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	ASSERT(args->size() < 2)
	auto dir_list_ = PyList::create();
	if (dir_list_.is_err()) return dir_list_;
	auto *dir_list = dir_list_.unwrap_as<PyList>();
	if (args->size() == 0) {
		for (const auto &[k, _] : interpreter.execution_frame()->locals()->map()) {
			auto obj_ = PyObject::from(k);
			if (obj_.is_err()) return obj_;
			dir_list->elements().push_back(obj_.unwrap_as<PyObject>());
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
			auto object_ = PyObject::from(arg);
			if (object_.is_err()) return object_;
			auto *object = object_.unwrap_as<PyObject>();
			for (const auto &[k, _] : object->attributes().map()) {
				dir_list->elements().push_back(k);
			}
		}
	}

	dir_list->sort();
	return dir_list_;
}

PyResult repr(const PyTuple *args, const PyDict *, Interpreter &)
{
	if (args->size() != 1) {
		return PyResult::Err(
			type_error("repr() takes exactly one argument ({} given)", args->size()));
	}
	return PyObject::from(args->elements()[0]).and_then<PyObject>([](auto *obj) {
		return obj->__repr__();
	});
}

PyResult abs(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (args->size() != 1) {
		return PyResult::Err(
			type_error("abs() takes exactly one argument ({} given)", args->size()));
	}
	if (kwargs && !kwargs->map().empty()) {
		return PyResult::Err(type_error("abs() takes no keyword arguments"));
	}
	return PyObject::from(args->elements()[0]).and_then<PyObject>([](auto *obj) {
		return obj->abs();
	});
}

PyResult staticmethod(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (args->size() != 1) {
		return PyResult::Err(
			type_error("staticmethod() takes exactly one argument ({} given)", args->size()));
	}
	if (kwargs && !kwargs->map().empty()) {
		return PyResult::Err(type_error("staticmethod() takes no keyword arguments"));
	}

	auto object_ = PyObject::from(args->elements()[0]);
	if (object_.is_err()) return object_;
	auto *function = object_.unwrap_as<PyFunction>();

	return PyStaticMethod::create(function->function_name(), function);
}

PyResult isinstance(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (args->size() != 2) {
		return PyResult::Err(type_error("isinstance expected 2 arguments, got {}", args->size()));
	}

	if (kwargs && !kwargs->map().empty()) {
		return PyResult::Err(type_error("isinstance() takes no keyword arguments"));
	}
	auto object_ = PyObject::from(args->elements()[0]);
	if (object_.is_err()) return object_;
	auto *object = object_.unwrap_as<PyObject>();
	auto classinfo_ = PyObject::from(args->elements()[1]);
	if (classinfo_.is_err()) return classinfo_;
	auto *classinfo = classinfo_.unwrap_as<PyObject>();

	if (auto *class_info_tuple = as<PyTuple>(classinfo)) {
		(void)class_info_tuple;
		TODO();
	} else if (auto *class_info_type = as<PyType>(classinfo)) {
		if (object->type() == class_info_type) {
			return PyResult::Ok(py_true());
		} else {
			auto mro_ = object->type()->mro();
			if (mro_.is_err()) { return mro_; }
			auto *mro = mro_.unwrap_as<PyList>();
			for (const auto &m : mro->elements()) {
				if (std::get<PyObject *>(m) == class_info_type) { return PyResult::Ok(py_true()); }
			}
			return PyResult::Ok(py_false());
		}
	} else {
		TODO();
	}
}

PyResult issubclass(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (args->size() != 2) {
		return PyResult::Err(type_error("issubclass expected 2 arguments, got {}", args->size()));
	}

	if (kwargs && !kwargs->map().empty()) {
		return PyResult::Err(type_error("issubclass() takes no keyword arguments"));
	}
	auto c = PyObject::from(args->elements()[0]);
	if (c.is_err()) return c;
	auto *class_ = c.unwrap_as<PyObject>();
	auto classinfo_ = PyObject::from(args->elements()[1]);
	if (classinfo_.is_err()) return classinfo_;
	auto *classinfo = classinfo_.unwrap_as<PyObject>();

	if (auto *class_info_tuple = as<PyTuple>(classinfo)) {
		(void)class_info_tuple;
		TODO();
	} else if (auto *class_info_type = as<PyType>(classinfo)) {
		auto *class_as_type = as<PyType>(class_);
		if (!class_as_type) {
			return PyResult::Err(type_error("issubclass() arg 1 must be a class"));
		}
		return PyResult::Ok(class_as_type->issubclass(class_info_type) ? py_true() : py_false());
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

	s_builtin_module =
		heap.allocate<PyModule>(PyString::create("__builtins__").unwrap_as<PyString>());

	for (auto *type : types) {
		s_builtin_module->insert(PyString::create(type->name()).unwrap_as<PyString>(), type);
	}

	initialize_exceptions(s_builtin_module);

	s_builtin_module->insert(PyString::create("__build_class__").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>(
			"__build_class__", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return build_class(args, kwargs, interpreter);
			}));

	s_builtin_module->insert(PyString::create("abs").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("abs", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return abs(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("dir").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("dir", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return dir(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("getattr").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("getattr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return getattr(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("globals").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("globals", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return globals(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("hasattr").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("hasattr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return hasattr(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("hex").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("hex", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return hex(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("id").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("id", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return id(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("iter").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("iter", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return iter(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("isinstance").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>(
			"isinstance", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return isinstance(args, kwargs, interpreter);
			}));

	s_builtin_module->insert(PyString::create("issubclass").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>(
			"issubclass", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return issubclass(args, kwargs, interpreter);
			}));

	s_builtin_module->insert(PyString::create("locals").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("locals", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return locals(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("len").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("len", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return len(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("next").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("next", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return next(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("ord").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("ord", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return ord(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("print").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("print", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return print(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("repr").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("repr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return repr(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("setattr").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>("setattr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return setattr(args, kwargs, interpreter);
		}));

	s_builtin_module->insert(PyString::create("staticmethod").unwrap_as<PyString>(),
		heap.allocate<PyNativeFunction>(
			"staticmethod", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return staticmethod(args, kwargs, interpreter);
			}));

	return s_builtin_module;
}

}// namespace py