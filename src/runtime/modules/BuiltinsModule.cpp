#include "Modules.hpp"
#include "runtime/AssertionError.hpp"
#include "runtime/AttributeError.hpp"
#include "runtime/Import.hpp"
#include "runtime/ImportError.hpp"
#include "runtime/IndexError.hpp"
#include "runtime/KeyError.hpp"
#include "runtime/LookupError.hpp"
#include "runtime/ModuleNotFoundError.hpp"
#include "runtime/NameError.hpp"
#include "runtime/NotImplementedError.hpp"
#include "runtime/OSError.hpp"
#include "runtime/PyArgParser.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyBytes.hpp"
#include "runtime/PyCell.hpp"
#include "runtime/PyCode.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyRange.hpp"
#include "runtime/PyStaticMethod.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"
#include "runtime/RuntimeError.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/SyntaxError.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/UnboundLocalError.hpp"
#include "runtime/Value.hpp"
#include "runtime/ValueError.hpp"
#include "runtime/modules/Modules.hpp"
#include "runtime/types/builtin.hpp"
#include "runtime/warnings/ImportWarning.hpp"
#include "runtime/warnings/Warning.hpp"

#include "executable/Mangler.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/Bytecode.hpp"
#include "executable/bytecode/codegen/BytecodeGenerator.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"

#include "interpreter/Interpreter.hpp"

#include "lexer/Lexer.hpp"

#include "memory/GarbageCollector.hpp"

#include "parser/Parser.hpp"

#include "vm/VM.hpp"

#include "utilities.hpp"
#include <algorithm>
#include <variant>

using namespace py;

static PyModule *s_builtin_module = nullptr;

namespace {

PyResult<PyObject *> print(const PyTuple *args, const PyDict *kwargs, Interpreter &)
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
				return Err(type_error(
					"sep must be None or a string, not {}", obj.unwrap()->type()->name()));
			}
			separator = std::get<String>(maybe_str).s;
		}
		if (auto it = kwargs->map().find(end_keyword); it != kwargs->map().end()) {
			auto maybe_str = it->second;
			if (!std::holds_alternative<String>(maybe_str)) {
				auto obj =
					std::visit([](const auto &value) { return PyObject::from(value); }, maybe_str);
				if (obj.is_err()) return obj;
				return Err(type_error(
					"end must be None or a string, not {}", obj.unwrap()->type()->name()));
			}
			end = std::get<String>(maybe_str).s;
		}
	}
	auto strfunc = [](const PyResult<PyObject *> &arg) -> PyResult<PyString *> {
		if (arg.is_err()) return Err(arg.unwrap_err());
		return arg.unwrap()->str();
	};

	auto arg_it = args->begin();
	auto arg_it_end = args->end();
	if (arg_it == arg_it_end) {
		std::cout << std::endl;
		return Ok(py_none());
	}
	--arg_it_end;

	while (arg_it != arg_it_end) {
		spdlog::debug("arg function ptr: {}", static_cast<void *>((*arg_it).unwrap()));
		auto reprobj_ = strfunc(*arg_it);
		if (reprobj_.is_err()) { return reprobj_; }
		auto reprobj = reprobj_.unwrap();
		spdlog::debug("repr result: {}", reprobj->value());
		std::cout << reprobj->value() << separator;
		std::advance(arg_it, 1);
	}

	spdlog::debug("arg function ptr: {}", static_cast<void *>((*arg_it).unwrap()));
	auto reprobj_ = strfunc(*arg_it);
	if (reprobj_.is_err()) { return reprobj_; }
	auto reprobj = reprobj_.unwrap();
	spdlog::debug("repr result: {}", reprobj->value());
	std::cout << reprobj->value() << end;

	return Ok(py_none());
}


PyResult<PyObject *> iter(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	ASSERT(args->size() == 1)
	const auto &arg = args->operator[](0);
	if (kwargs) { return Err(type_error("iter() takes no keyword arguments")); }
	return arg.and_then([](auto *obj) { return obj->iter(); });
}

PyResult<PyObject *> hash(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	ASSERT(args->size() == 1)
	const auto &arg = args->operator[](0);
	if (kwargs) { return Err(type_error("hash() takes no keyword arguments")); }
	return arg.and_then([](auto *obj) { return obj->hash(); }).and_then([](const size_t h) {
		return PyInteger::create(h);
	});
}

PyResult<PyObject *> next(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	ASSERT(args->size() == 1)
	if (kwargs) { return Err(type_error("next() takes no keyword arguments")); }
	const auto &arg = args->operator[](0);
	return arg.and_then([](auto *obj) { return obj->next(); });
}


PyResult<PyObject *>
	build_class(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	if (args->size() < 2) {
		return Err(type_error("__build_class__: not enough arguments, got {}", args->size()));
	}
	bool metaclass_is_class = false;
	auto metaclass_ = [kwargs, &metaclass_is_class]() -> PyResult<PyObject *> {
		if (kwargs && kwargs->map().size() > 0) {
			auto it = kwargs->map().find(String{ "metaclass" });
			if (it != kwargs->map().end()) {
				return PyObject::from(it->second).and_then([&metaclass_is_class](PyObject *obj) {
					if (obj->type()->issubclass(py::types::type())) { metaclass_is_class = true; }
					return Ok(obj);
				});
			}
		}
		return Ok(py_none());
	}();
	if (metaclass_.is_err()) return metaclass_;
	auto *metaclass = metaclass_.unwrap();
	auto maybe_function_location_ = args->operator[](0);
	if (maybe_function_location_.is_err()) return maybe_function_location_;
	auto *maybe_function_location = maybe_function_location_.unwrap();
	auto mangled_class_name_ = args->operator[](1);
	if (mangled_class_name_.is_err()) return mangled_class_name_;
	auto *mangled_class_name = mangled_class_name_.unwrap();
	spdlog::debug("__build_class__({}, {})",
		mangled_class_name->to_string(),
		maybe_function_location->to_string());

	if (!as<PyString>(mangled_class_name)) {
		return Err(type_error("__build_class__: name is not a string"));
	}

	const auto mangled_class_name_as_string = as<PyString>(mangled_class_name)->value();

	PyResult<PyFunction *> callable = [&]() -> PyResult<PyFunction *> {
		// TODO: Remove as<PyInteger>(maybe_function_location) branch. This is deprecated
		if (as<PyInteger>(maybe_function_location)) {
			// auto function_id = std::get<int64_t>(pynumber->value().value);
			// FIXME: what should be the global dictionary for this?
			// FIXME: what should be the module for this?
			auto *f = interpreter.execution_frame()->code()->make_function(
				mangled_class_name_as_string, {}, {}, {});
			ASSERT(as<PyFunction>(f))
			return Ok(as<PyFunction>(f));
		} else if (auto *pyfunc = as<PyFunction>(maybe_function_location)) {
			return Ok(pyfunc);
		} else {
			return Err(type_error("__build_class__: func must be callable"));
		}
	}();

	if (callable.is_err()) { TODO(); }

	const std::string class_name_str =
		Mangler::default_mangler().class_demangle(mangled_class_name_as_string);

	auto class_name_ = PyString::create(class_name_str);
	if (class_name_.is_err()) { return Err(class_name_.unwrap_err()); }
	auto *class_name = class_name_.unwrap();

	std::vector<Value> bases_vector;
	if (args->size() > 2) {
		bases_vector.reserve(args->size() - 2);
		auto it = args->elements().begin() + 2;
		while (it != args->elements().end()) {
			bases_vector.push_back(*it);
			it++;
		}
	}

	auto bases_ = PyTuple::create(bases_vector);
	if (bases_.is_err()) { return Err(bases_.unwrap_err()); }
	auto *bases = bases_.unwrap();

	// finalize this class' metaclass
	if (metaclass == py_none()) {
		if (bases->size() == 0) {
			// if there are no bases, use `type`
			metaclass = py::types::type();
		} else {
			// else get the type of the first base
			metaclass = PyObject::from(bases->elements()[0]).unwrap()->type();
		}
		metaclass_is_class = true;
	}

	if (metaclass_is_class) {
		std::vector<PyType *> bases_vector;
		for (const auto &base : bases->elements()) {
			auto base_ = PyObject::from(base);
			if (base_.is_err()) { return base_; }
			if (auto *b = as<PyType>(base_.unwrap())) {
				bases_vector.push_back(b);
			} else {
				return Err(type_error("bases must be types"));
			}
		}
		auto winner = PyType::calculate_metaclass(static_cast<PyType *>(metaclass), bases_vector);
		if (winner.is_err()) { return Err(winner.unwrap_err()); }
		metaclass = const_cast<PyType *>(winner.unwrap());
	}

	// lookup __prepare__ and instantiate namespace
	auto ns_ = [metaclass, class_name, bases, kwargs]() -> PyResult<PyObject *> {
		if (metaclass == types::type()) {
			return PyDict::create();
		} else {
			auto prepare = PyString::create("__prepare__");
			if (prepare.is_err()) { return prepare; }
			auto new_kwargs_ = [kwargs]() {
				if (kwargs && !kwargs->map().empty()) {
					auto prepare_kwargs = kwargs->map();
					prepare_kwargs.erase(String{ "metaclass" });
					return PyDict::create(prepare_kwargs);
				} else {
					return PyDict::create();
				}
			}();
			if (new_kwargs_.is_err()) { return new_kwargs_; }
			auto *new_kwargs = new_kwargs_.unwrap();
			auto result = metaclass->lookup_attribute(prepare.unwrap());
			if (std::get<0>(result).is_ok() && std::get<1>(result) == LookupAttrResult::FOUND) {
				return std::get<0>(result).and_then(
					[class_name, bases, new_kwargs](PyObject *prepare) {
						auto args = PyTuple::create(class_name, bases);
						return prepare->call(args.unwrap(), new_kwargs);
					});
			} else {
				return PyDict::create();
			}
		}
	}();

	if (ns_.is_err()) { return Err(ns_.unwrap_err()); }
	auto *ns = ns_.unwrap();

	// this calls a function that defines a class
	// For example:
	// class A:
	//   def foo(self):
	//     pass
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
	auto *empty_args = args_.unwrap();

	auto kwargs_ = PyDict::create();
	if (kwargs_.is_err()) { return kwargs_; }
	auto *empty_kwargs = kwargs_.unwrap();
	auto classcell = callable.unwrap()->call_with_frame(ns, empty_args, empty_kwargs);
	if (classcell.is_err()) { return classcell; }

	auto call_args = PyTuple::create(class_name, bases, ns);
	if (call_args.is_err()) { return Err(call_args.unwrap_err()); }

	auto cls = metaclass->call(call_args.unwrap(), nullptr);

	// FIXME: according to CPython this is *not* how you do it, but RustPython does it this way
	//        what are the implications? Find out how CPython sets __classcell__.
	return cls.and_then([&classcell](PyObject *cls) {
		if (as<PyCell>(classcell.unwrap())) { as<PyCell>(classcell.unwrap())->set_cell(cls); }
		return Ok(cls);
	});
}

PyResult<PyObject *> globals(const PyTuple *, const PyDict *, Interpreter &interpreter)
{
	return Ok(interpreter.execution_frame()->globals());
}


PyResult<PyObject *> locals(const PyTuple *, const PyDict *, Interpreter &interpreter)
{
	return Ok(interpreter.execution_frame()->locals());
}


PyResult<PyObject *> len(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (args->size() != 1) {
		return Err(type_error("len() takes exactly one argument ({} given)", args->size()));
	}
	if (kwargs && !kwargs->map().empty()) {
		return Err(type_error("len() takes no keyword arguments"));
	}

	return PyObject::from(args->elements()[0]).and_then([](PyObject *o) -> PyResult<PyObject *> {
		auto mapping = o->as_mapping();
		if (mapping.is_err()) { return Err(mapping.unwrap_err()); }
		if (auto r = mapping.unwrap().len(); r.is_ok()) {
			return PyInteger::create(r.unwrap());
		} else {
			return Err(r.unwrap_err());
		}
	});
}

PyResult<PyObject *> id(const PyTuple *args, const PyDict *, Interpreter &)
{
	ASSERT(args->size() == 1)
	auto obj = args->operator[](0);
	if (obj.is_err()) return obj;
	return PyInteger::create(bit_cast<intptr_t>(obj.unwrap()));
}

PyResult<PyObject *> import(const PyTuple *args, const PyDict *, Interpreter &)
{
	// TODO: support globals, locals, fromlist and level
	ASSERT(args->size() > 0)
	auto arg0 = args->operator[](0);
	if (arg0.is_err()) return arg0;
	auto *name = arg0.unwrap();

	if (!as<PyString>(name)) {
		return Err(
			type_error("__import__(): name must be a string, not {}", name->type()->to_string()));
	}

	auto arg1 = [args]() -> PyResult<PyObject *> {
		if (args->size() > 1) {
			auto arg1 = args->operator[](1);
			if (arg1.is_err()) return arg1;
			auto *globals = arg1.unwrap();
			if (!as<PyDict>(globals) && globals != py_none()) {
				return Err(type_error("__import__(): globals must be a dict or None, not {}",
					globals->type()->to_string()));
			}
			return Ok(globals);
		} else {
			return Ok(py_none());
		}
	}();
	if (arg1.is_err()) return arg1;
	auto *globals = arg1.unwrap();

	auto arg2 = [args]() -> PyResult<PyObject *> {
		if (args->size() > 2) {
			auto arg2 = args->operator[](2);
			if (arg2.is_err()) return arg2;
			auto *locals = arg2.unwrap();
			return Ok(locals);
		} else {
			return Ok(py_none());
		}
	}();

	if (arg2.is_err()) return arg2;
	auto *locals = arg2.unwrap();

	auto arg3 = [args]() -> PyResult<PyObject *> {
		if (args->size() > 3) {
			auto arg3 = args->operator[](3);
			if (arg3.is_err()) return arg3;
			auto *fromlist = arg3.unwrap();
			return Ok(fromlist);
		} else {
			return PyTuple::create();
		}
	}();

	if (arg3.is_err()) return arg3;
	auto *fromlist = arg3.unwrap();

	auto arg4 = [args]() -> PyResult<PyObject *> {
		if (args->size() > 1) {
			auto arg4 = args->operator[](4);
			if (arg4.is_err()) return arg4;
			auto *level = arg4.unwrap();
			if (!as<PyInteger>(level)) {
				return Err(type_error(
					"__import__(): level must be an int, not {}", level->type()->to_string()));
			}
			return Ok(level);
		} else {
			return PyInteger::create(0);
		}
	}();
	if (arg4.is_err()) return arg4;
	auto *level = arg4.unwrap();


	return import_module_level_object(as<PyString>(name),
		as<PyDict>(globals),
		locals,
		fromlist,
		static_cast<uint32_t>(as<PyInteger>(level)->as_size_t()));
}

PyResult<PyObject *> hasattr(const PyTuple *args, const PyDict *, Interpreter &)
{
	if (args->size() != 2) {
		return Err(type_error("hasattr expected 2 arguments, got {}", args->size()));
	}
	auto obj_ = PyObject::from(args->elements()[0]);
	if (obj_.is_err()) return obj_;
	auto *obj = obj_.unwrap();
	auto name_ = PyObject::from(args->elements()[1]);
	if (name_.is_err()) return name_;
	auto *name = name_.unwrap();
	if (!as<PyString>(name)) { return Err(type_error("hasattr(): attribute name must be string")); }

	auto [result, found_status] = obj->lookup_attribute(name);
	if (found_status == LookupAttrResult::FOUND) {
		return Ok(py_true());
	} else if (found_status == LookupAttrResult::NOT_FOUND) {
		return Ok(py_false());
	} else {
		return result;
	}
}

PyResult<PyObject *> getattr(const PyTuple *args, const PyDict *, Interpreter &)
{
	if (args->size() != 2 && args->size() != 3) {
		return Err(type_error("getattr expected 2 or 3 arguments, got {}", args->size()));
	}
	auto obj_ = PyObject::from(args->elements()[0]);
	if (obj_.is_err()) return obj_;
	auto *obj = obj_.unwrap();
	auto name_ = PyObject::from(args->elements()[1]);
	if (name_.is_err()) return name_;
	auto *name = name_.unwrap();
	if (!as<PyString>(name)) { return Err(type_error("getattr(): attribute name must be string")); }

	if (args->size() == 2) {
		auto result = obj->getattribute(name);
		if (result.is_ok()) { ASSERT(result.unwrap()); }
		return result;
	} else {
		auto default_value_ = PyObject::from(args->elements()[2]);
		if (default_value_.is_err()) return default_value_;
		auto *default_value = default_value_.unwrap();

		auto [attr_value, found_status] = obj->lookup_attribute(name);

		if (attr_value.is_err()) { return attr_value; }

		if (found_status == LookupAttrResult::FOUND) {
			if (attr_value.is_ok()) { ASSERT(attr_value.unwrap()); }
			return attr_value;
		} else {
			ASSERT(default_value);
			return Ok(default_value);
		}
	}
}

PyResult<PyObject *> setattr(const PyTuple *args, const PyDict *, Interpreter &)
{
	if (args->size() != 3) {
		return Err(type_error("setattr expected 3 arguments, got {}", args->size()));
	}
	auto obj_ = PyObject::from(args->elements()[0]);
	if (obj_.is_err()) return obj_;
	auto *obj = obj_.unwrap();
	auto name_ = PyObject::from(args->elements()[1]);
	if (name_.is_err()) return name_;
	auto *name = name_.unwrap();
	auto value_ = PyObject::from(args->elements()[2]);
	if (value_.is_err()) return value_;
	auto *value = value_.unwrap();

	if (!as<PyString>(name)) { return Err(type_error("setattr(): attribute name must be string")); }

	if (auto result = obj->setattribute(name, value); result.is_ok()) {
		return Ok(py_none());
	} else {
		return Err(result.unwrap_err());
	}
}

PyResult<PyObject *> hex(const PyTuple *args, const PyDict *, Interpreter &)
{
	ASSERT(args->size() == 1)
	auto obj_ = args->operator[](0);
	if (obj_.is_err()) return obj_;
	auto *obj = obj_.unwrap();
	if (auto pynumber = PyNumber::as_number(obj)) {
		if (std::holds_alternative<BigIntType>(pynumber->value().value)) {
			std::ostringstream os;
			os << std::hex << std::ios::showbase << std::get<BigIntType>(pynumber->value().value);
			return PyString::create(os.str());
		} else {
			// FIXME: when float is separated from integer fix this
			return Err(type_error(
				"'float' object cannot be interpreted as an integer", obj->type()->name()));
		}
	} else {
		return Err(
			type_error("'{}' object cannot be interpreted as an integer", obj->type()->name()));
	}
}

PyResult<PyObject *> ord(const PyTuple *args, const PyDict *, Interpreter &)
{
	ASSERT(args->size() == 1)
	auto obj_ = args->operator[](0);
	if (obj_.is_err()) return obj_;
	auto *obj = obj_.unwrap();
	if (auto pystr = as<PyString>(obj)) {
		if (auto codepoint = pystr->codepoint()) {
			return PyObject::from(Number{ static_cast<int64_t>(*codepoint) });
		} else {
			auto mapping = pystr->as_mapping();
			if (mapping.is_err()) { return Err(mapping.unwrap_err()); }
			auto size = mapping.unwrap().len();
			if (size.is_err()) { return Err(size.unwrap_err()); }
			return Err(type_error(
				"ord() expected a character, but string of length {} found", size.unwrap()));
		}
	} else {
		return Err(
			type_error("ord() expected string of length 1, but {} found", obj->type()->name()));
	}
}

PyResult<PyObject *> chr(const PyTuple *args, const PyDict *, Interpreter &)
{
	ASSERT(args->size() == 1)
	auto obj_ = args->operator[](0);
	if (obj_.is_err()) return obj_;
	auto *obj = obj_.unwrap();

	if (auto cp = PyNumber::as_number(obj)) {
		if (std::holds_alternative<double>(cp->value().value)) {
			return Err(type_error("'float' object cannot be interpreted as an integer"));
		}
		return PyString::chr(std::get<BigIntType>(cp->value().value));
	} else {
		return Err(
			type_error("'{}' object cannot be interpreted as an integer", obj->type()->name()));
	}
}

PyResult<PyObject *> dir(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	ASSERT(args->size() < 2)
	auto dir_list_ = PyList::create();
	if (dir_list_.is_err()) return Err(dir_list_.unwrap_err());
	auto *dir_list = dir_list_.unwrap();
	if (args->size() == 0) {
		ASSERT(as<PyDict>(interpreter.execution_frame()->locals()));
		for (const auto &[k, _] : as<PyDict>(interpreter.execution_frame()->locals())->map()) {
			auto obj_ = PyObject::from(k);
			if (obj_.is_err()) return obj_;
			dir_list->elements().push_back(obj_.unwrap());
		}
	} else {
		const auto &arg = args->elements()[0];

		// If the object is a module object, the list contains the names of the module’s attributes.
		if (std::holds_alternative<PyObject *>(arg) && as<PyModule>(std::get<PyObject *>(arg))) {
			auto *pymodule = as<PyModule>(std::get<PyObject *>(arg));
			for (const auto &[k, _] : pymodule->symbol_table()->map()) {
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
			auto *object = object_.unwrap();
			for (const auto &[k, _] : object->attributes()->map()) {
				dir_list->elements().push_back(k);
			}
		}
	}

	dir_list->sort();
	return Ok(static_cast<PyObject *>(dir_list_.unwrap()));
}

PyResult<PyObject *> repr(const PyTuple *args, const PyDict *, Interpreter &)
{
	if (args->size() != 1) {
		return Err(type_error("repr() takes exactly one argument ({} given)", args->size()));
	}
	return PyObject::from(args->elements()[0]).and_then([](auto *obj) { return obj->repr(); });
}

PyResult<PyObject *> abs(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (args->size() != 1) {
		return Err(type_error("abs() takes exactly one argument ({} given)", args->size()));
	}
	if (kwargs && !kwargs->map().empty()) {
		return Err(type_error("abs() takes no keyword arguments"));
	}
	return PyObject::from(args->elements()[0]).and_then([](auto *obj) { return obj->abs(); });
}

PyResult<PyObject *> max(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	if (!args || args->size() == 0) { return Err(type_error("")); }

	if (kwargs && kwargs->size() > 0) { TODO(); }

	if (args->size() == 1) {
		auto iterable = PyObject::from(args->elements()[0]);
		if (iterable.is_err()) return Err(iterable.unwrap_err());

		auto iterator = iterable.unwrap()->iter();
		if (iterator.is_err()) return Err(iterator.unwrap_err());

		auto value = iterator.unwrap()->next();
		if (value.is_err()) return value;
		auto *max_value = value.unwrap();

		while (value.is_ok()) {
			auto cmp = value.unwrap()->richcompare(max_value, RichCompare::Py_GT);
			if (cmp.is_err()) return cmp;
			if (cmp.unwrap() == py_true()) { max_value = value.unwrap(); }
			value = iterator.unwrap()->next();
		}

		if (value.unwrap_err()->type() != stop_iteration()->type()) {
			return Err(value.unwrap_err());
		}

		return Ok(max_value);
	} else {
		std::optional<Value> max_value;
		for (const auto &el : args->elements()) {
			if (max_value.has_value()) {
				auto cmp = greater_than(el, *max_value, interpreter);
				if (cmp.is_err()) return Err(cmp.unwrap_err());
				auto r = truthy(cmp.unwrap(), interpreter);
				if (r.is_err()) return Err(r.unwrap_err());
				if (r.unwrap()) { max_value = el; }
			} else {
				max_value = el;
			}
		}

		ASSERT(max_value.has_value());

		return PyObject::from(*max_value);
	}
}

PyResult<PyObject *> min(const PyTuple *args, const PyDict *kwargs, Interpreter &interpreter)
{
	if (!args || args->size() == 0) { return Err(type_error("")); }

	if (kwargs && kwargs->size() > 0) { TODO(); }

	if (args->size() == 1) {
		auto iterable = PyObject::from(args->elements()[0]);
		if (iterable.is_err()) return Err(iterable.unwrap_err());

		auto iterator = iterable.unwrap()->iter();
		if (iterator.is_err()) return Err(iterator.unwrap_err());

		auto value = iterator.unwrap()->next();
		if (value.is_err()) return value;
		auto *min_value = value.unwrap();

		while (value.is_ok()) {
			auto cmp = value.unwrap()->richcompare(min_value, RichCompare::Py_LT);
			if (cmp.is_err()) return cmp;
			if (cmp.unwrap() == py_true()) { min_value = value.unwrap(); }
			value = iterator.unwrap()->next();
		}

		if (value.unwrap_err()->type() != stop_iteration()->type()) {
			return Err(value.unwrap_err());
		}

		return Ok(min_value);
	} else {
		std::optional<Value> min_value;
		for (const auto &el : args->elements()) {
			if (min_value.has_value()) {
				auto cmp = less_than(el, *min_value, interpreter);
				if (cmp.is_err()) return Err(cmp.unwrap_err());
				auto r = truthy(cmp.unwrap(), interpreter);
				if (r.is_err()) return Err(r.unwrap_err());
				if (r.unwrap()) { min_value = el; }
			} else {
				min_value = el;
			}
		}

		ASSERT(min_value.has_value());

		return PyObject::from(*min_value);
	}
}

PyResult<PyObject *> isinstance(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (args->size() != 2) {
		return Err(type_error("isinstance expected 2 arguments, got {}", args->size()));
	}

	if (kwargs && !kwargs->map().empty()) {
		return Err(type_error("isinstance() takes no keyword arguments"));
	}
	auto object_ = PyObject::from(args->elements()[0]);
	if (object_.is_err()) return object_;
	auto *object = object_.unwrap();
	auto classinfo_ = PyObject::from(args->elements()[1]);
	if (classinfo_.is_err()) return classinfo_;
	auto *classinfo = classinfo_.unwrap();

	std::vector<PyType *> types;
	if (auto *class_info_tuple = as<PyTuple>(classinfo)) {
		types.reserve(class_info_tuple->elements().size());
		for (const auto &el : class_info_tuple->elements()) {
			auto el_obj = PyObject::from(el);
			if (el_obj.is_err()) return el_obj;
			if (!as<PyType>(el_obj.unwrap())) {
				return Err(type_error("isinstance() arg 2 must be a type or tuple of types"));
			}
			types.push_back(as<PyType>(el_obj.unwrap()));
		}
	} else if (auto *class_info_type = as<PyType>(classinfo)) {
		types.push_back(class_info_type);
	} else {
		return Err(type_error("isinstance() arg 2 must be a type or tuple of types"));
	}

	const auto result = std::any_of(types.begin(), types.end(), [object](PyType *const &t) {
		return object->type()->issubclass(t);
	});

	return Ok(result ? py_true() : py_false());
}

PyResult<PyObject *> issubclass(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (args->size() != 2) {
		return Err(type_error("issubclass expected 2 arguments, got {}", args->size()));
	}

	if (kwargs && !kwargs->map().empty()) {
		return Err(type_error("issubclass() takes no keyword arguments"));
	}
	auto c = PyObject::from(args->elements()[0]);
	if (c.is_err()) return c;
	auto *class_ = c.unwrap();
	auto classinfo_ = PyObject::from(args->elements()[1]);
	if (classinfo_.is_err()) return classinfo_;
	auto *classinfo = classinfo_.unwrap();

	auto *class_as_type = as<PyType>(class_);
	if (!class_as_type) { return Err(type_error("issubclass() arg 1 must be a class")); }

	if (auto *class_info_tuple = as<PyTuple>(classinfo)) {
		(void)class_info_tuple;
		TODO();
	} else if (auto *class_info_type = as<PyType>(classinfo)) {
		return Ok(class_as_type->issubclass(class_info_type) ? py_true() : py_false());
	} else {
		return Err(type_error("issubclass() arg 2 must be a class or tuple of classes"));
	}
}

PyResult<PyObject *> all(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (args->size() != 1) {
		return Err(type_error("all expected 1 arguments, got {}", args->size()));
	}

	if (kwargs && !kwargs->map().empty()) {
		return Err(type_error("all() takes no keyword arguments"));
	}
	auto iterable_ = PyObject::from(args->elements()[0]);
	if (iterable_.is_err()) return iterable_;
	auto *iterable = iterable_.unwrap();

	const auto &iterator = iterable->iter();
	if (iterator.is_err()) return iterator;
	auto next_value = iterator.unwrap()->next();
	while (!next_value.is_err()) {
		const auto is_truthy = next_value.unwrap()->true_();
		if (is_truthy.is_err()) return Err(is_truthy.unwrap_err());
		if (!is_truthy.unwrap()) { return Ok(py_false()); }
		next_value = iterator.unwrap()->next();
	}

	if (next_value.unwrap_err()->type()->issubclass(types::stop_iteration())) {
		return Ok(py_true());
	} else {
		return next_value;
	}
}


PyResult<PyObject *> any(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (args->size() != 1) {
		return Err(type_error("any expected 1 arguments, got {}", args->size()));
	}

	if (kwargs && !kwargs->map().empty()) {
		return Err(type_error("any() takes no keyword arguments"));
	}
	auto iterable_ = PyObject::from(args->elements()[0]);
	if (iterable_.is_err()) return iterable_;
	auto *iterable = iterable_.unwrap();

	const auto &iterator = iterable->iter();
	if (iterator.is_err()) return iterator;
	auto next_value = iterator.unwrap()->next();
	while (!next_value.is_err()) {
		const auto is_truthy = next_value.unwrap()->true_();
		if (is_truthy.is_err()) return Err(is_truthy.unwrap_err());
		if (is_truthy.unwrap()) { return Ok(py_true()); }
		next_value = iterator.unwrap()->next();
	}

	if (next_value.unwrap_err()->type()->issubclass(types::stop_iteration())) {
		return Ok(py_false());
	} else {
		return next_value;
	}
}

PyResult<PyObject *> exec(const PyTuple *args, const PyDict *, Interpreter &interpreter)
{
	ASSERT(args)
	if (args->size() < 1) {
		return Err(type_error("exec expected at least 1 argument, got {}", args->size()));
	}
	if (args->size() > 3) {
		return Err(type_error("exec expected at most 3 arguments, got {}", args->size()));
	}

	auto source_ = PyObject::from(args->elements()[0]);
	auto globals_ = args->size() >= 2 ? PyObject::from(args->elements()[1]) : Ok(py_none());
	auto locals_ = args->size() == 3 ? PyObject::from(args->elements()[2]) : Ok(py_none());

	if (source_.is_err()) return source_;
	if (globals_.is_err()) return globals_;
	if (locals_.is_err()) return locals_;

	auto *source = source_.unwrap();
	auto *globals = globals_.unwrap();
	auto *locals = locals_.unwrap();

	ASSERT(source);
	ASSERT(globals);
	ASSERT(locals);

	if (globals == py_none()) {
		globals = interpreter.execution_frame()->globals();
		if (locals == py_none()) { locals = interpreter.execution_frame()->locals(); }
		if (!globals || !locals) { TODO(); }
	} else if (locals == py_none()) {
		locals = globals;
	}

	if (!as<PyDict>(globals)) {
		return Err(type_error("exec() globals must be a dict, not {}", globals->type()->name()));
	}

	if (locals->as_mapping().is_err() && locals != py_none()) {
		return Err(type_error("locals must be a mapping or None, not {}", locals->type()->name()));
	}

	if (!as<PyDict>(globals)->map().contains(String{ "__builtin__" })) {
		as<PyDict>(globals)->insert(
			String{ "__builtin__" }, interpreter.execution_frame()->builtins());
	}

	if (auto *code = as<PyCode>(source)) {
		if (!as<PyDict>(locals)) { TODO(); }
		return code->eval(as<PyDict>(globals),
			as<PyDict>(locals),
			PyTuple::create().unwrap(),
			PyDict::create().unwrap(),
			{},
			{},
			{},
			PyString::create("").unwrap());
	} else {
		TODO();
	}
}

PyResult<PyObject *> eval(PyTuple *args, PyDict *kwargs, Interpreter &interpreter)
{
	auto result = PyArgsParser<PyObject *, PyObject *, PyObject *>::unpack_tuple(args,
		kwargs,
		"eval",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 3>{},
		py_none() /* globals */,
		py_none() /* locals */);

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [source, globals, locals] = result.unwrap();

	if (globals == py_none()) {
		globals = interpreter.execution_frame()->globals();
		if (locals == py_none()) { locals = interpreter.execution_frame()->locals(); }
	} else if (locals == py_none()) {
		locals = globals;
	}

	if (!as<PyDict>(globals)) {
		return Err(type_error("eval() globals must be a dict, not {}", globals->type()->name()));
	}

	if (locals != py_none() && locals->as_mapping().is_err()) {
		return Err(type_error("locals must be a mapping"));
	}

	ASSERT(globals && globals != py_none() && locals && locals != py_none());

	if (!as<PyDict>(globals)->map().contains(String{ "__builtin__" })) {
		as<PyDict>(globals)->insert(
			String{ "__builtin__" }, interpreter.execution_frame()->builtins());
	}

	if (!as<PyDict>(locals)) { TODO(); }

	if (auto code = as<PyCode>(source)) {
		return code->eval(as<PyDict>(globals),
			as<PyDict>(locals),
			PyTuple::create().unwrap(),
			PyDict::create().unwrap(),
			{},
			{},
			{},
			PyString::create("").unwrap());
	} else if (auto s = as<PyString>(source)) {
		auto l = Lexer::create(s->value() + "\n", "");
		parser::Parser p{ l };
		auto m = p.parse_expression();
		if (m.is_err()) { return Err(m.unwrap_err()); }
		auto program = compiler::compile(
			m.unwrap(), {}, compiler::Backend::MLIR, compiler::OptimizationLevel::None);
		auto code = program->main_function();
		ASSERT(as<PyCode>(code));
		return as<PyCode>(code)->eval(as<PyDict>(globals),
			as<PyDict>(locals),
			PyTuple::create().unwrap(),
			PyDict::create().unwrap(),
			{},
			{},
			{},
			PyString::create("").unwrap());
	}

	TODO();
}

PyResult<PyObject *> compile(const PyTuple *args, const PyDict *, Interpreter &)
{
	ASSERT(args)
	if (args->size() < 1) {
		return Err(type_error("compile() missing required argument 'source' (pos 0)"));
	}
	auto arg0_ = PyObject::from(args->elements()[0]);
	if (arg0_.is_err()) return arg0_;
	auto *source = arg0_.unwrap();

	if (args->size() < 2) {
		return Err(type_error("compile() missing required argument 'filename' (pos 0)"));
	}
	auto arg1_ = PyObject::from(args->elements()[1]);
	if (arg1_.is_err()) return arg1_;
	auto *filename = arg1_.unwrap();

	if (args->size() < 3) {
		return Err(type_error("compile() missing required argument 'mode' (pos 0)"));
	}
	auto arg2_ = PyObject::from(args->elements()[2]);
	if (arg2_.is_err()) return arg2_;
	auto *mode = arg2_.unwrap();

	auto args3_ = [args]() -> PyResult<PyObject *> {
		if (args->size() < 4) return PyInteger::create(0);
		return PyObject::from(args->elements()[3]);
	}();
	if (args3_.is_err()) return args3_;
	auto *flags = args3_.unwrap();

	auto args4_ = [args]() -> PyResult<PyObject *> {
		if (args->size() < 5) return Ok(py_false());
		return PyObject::from(args->elements()[4]);
	}();
	if (args4_.is_err()) return args4_;
	auto *dont_inherit = args4_.unwrap();

	auto args5_ = [args]() -> PyResult<PyObject *> {
		if (args->size() < 6) return PyInteger::create(-1);
		return PyObject::from(args->elements()[5]);
	}();
	if (args5_.is_err()) return args5_;
	auto *optimize = args5_.unwrap();

	ASSERT(as<PyString>(source) || as<PyBytes>(source));
	ASSERT(as<PyString>(filename));
	ASSERT(as<PyString>(mode));
	ASSERT(as<PyInteger>(flags));
	ASSERT(as<PyBool>(dont_inherit));
	ASSERT(as<PyInteger>(optimize));

	auto source_str = [source]() {
		if (as<PyString>(source)) { return as<PyString>(source)->value(); }
		const auto &bytes = as<PyBytes>(source)->value().b;
		std::string source_str;
		source_str.reserve(bytes.size());
		std::transform(bytes.begin(),
			bytes.end(),
			std::back_inserter(source_str),
			[](const std::byte b) -> char { return static_cast<char>(b); });
		return source_str;
	}();
	const auto filename_str = as<PyString>(filename)->value();
	const auto mode_str = as<PyString>(mode)->value();
	if (mode_str == "exec") {
		if (source_str.back() != '\n') { source_str.append("\n"); }

		auto lexer = Lexer::create(source_str, filename_str);
		parser::Parser p{ lexer };
		p.parse();

		std::shared_ptr<Program> bytecode = compiler::compile(p.module(),
			{ filename_str },
			compiler::Backend::MLIR,
			compiler::OptimizationLevel::None);
		if (!bytecode) { TODO(); }

		return Ok(bytecode->main_function());
	} else if (mode_str == "eval") {
		TODO();
	} else if (mode_str == "single") {
		TODO();
	} else {
		return Err(value_error("compile() mode must be 'exec', 'eval' or 'single'"));
	}
}

PyResult<PyObject *> callable(const PyTuple *args, const PyDict *kwargs, Interpreter &)
{
	if (!args) {
		return Err(type_error("callable() takes exactly one argument (0 given)"));
	} else if (args->size() != 1) {
		return Err(type_error("callable() takes exactly one argument ({} given)", args->size()));
	}

	if (kwargs && kwargs->size() != 0) {
		return Err(type_error("callable() takes no keyword arguments", args->size()));
	}

	auto obj = args->elements()[0];
	return std::visit(overloaded{
						  [](auto) { return false; },
						  [](PyObject *obj) { return obj->type_prototype().__call__.has_value(); },
					  },
			   obj)
			   ? Ok(py_true())
			   : Ok(py_false());
}

PyResult<PyObject *> ascii(PyTuple *args, PyDict *kwargs, Interpreter &)
{
	auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
		kwargs,
		"ascii",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 1>{});

	if (result.is_err()) { return Err(result.unwrap_err()); }
	auto [obj] = result.unwrap();

	return PyString::convert_to_ascii(obj);
}

PyResult<PyObject *> sorted(PyTuple *args, PyDict *kwargs, Interpreter &interpreter)
{
	if (!args) { return Err(type_error("sorted expected 1 arguments, got 0")); }
	if (args->elements().size() != 1) {
		return Err(type_error("sorted expected 1 arguments, got {}", args->elements().size()));
	}

	auto iterable_ = PyObject::from(args->elements()[0]);
	if (iterable_.is_err()) { return iterable_; }
	auto *iterable = iterable_.unwrap();

	PyObject *key = py_none();
	bool reverse = false;

	if (kwargs) {
		if (auto it = kwargs->map().find(String{ "key" }); it != kwargs->map().end()) {
			auto key_ = PyObject::from(it->second);
			if (key_.is_err()) { return key_; }
			if (!key_.unwrap()->type_prototype().__call__.has_value()) {
				return Err(type_error("'{}' objects not callable", key_.unwrap()->type()->name()));
			}
			key = key_.unwrap();
		}
		if (auto it = kwargs->map().find(String{ "reverse" }); it != kwargs->map().end()) {
			auto t = truthy(it->second, interpreter);
			if (t.is_err()) { return Err(t.unwrap_err()); }
			reverse = t.unwrap();
		}
	}

	auto result_ = PyList::create();
	if (result_.is_err()) { return result_; }
	auto *result = result_.unwrap();

	PyList *cmp_list = nullptr;
	if (key != py_none()) {
		auto cmp_list_ = PyList::create();
		if (cmp_list_.is_err()) { return cmp_list_; }
		cmp_list = cmp_list_.unwrap();
	}

	auto iter_ = iterable->iter();
	if (iter_.is_err()) { return iter_; }
	auto *iter = iter_.unwrap();

	auto value = iter->next();
	while (value.is_ok()) {
		if (key != py_none()) {
			ASSERT(cmp_list);
			auto cmp_value = key->call(PyTuple::create(value.unwrap()).unwrap(), nullptr);
			if (cmp_value.is_err()) { return cmp_value; }
			cmp_list->elements().push_back(cmp_value.unwrap());
		}
		result->elements().push_back(value.unwrap());
		value = iter->next();
	}

	if (!value.unwrap_err()->type()->issubclass(types::stop_iteration())) { return value; }

	if (key == py_none()) {
		// FIXME: should throw exception when comparing, as returning true is
		// probably messing up the C++ Compare requirment
		PyResult<PyObject *> err = Ok(py_none());
		auto cmp = [&err](const Value &lhs, const Value &rhs) -> bool {
			if (auto cmp = less_than(lhs, rhs, VirtualMachine::the().interpreter()); cmp.is_ok()) {
				auto is_true = truthy(cmp.unwrap(), VirtualMachine::the().interpreter());
				if (is_true.is_err()) {
					err = Err(is_true.unwrap_err());
					return true;
				}
				return is_true.unwrap();
			} else {
				return false;
			}
		};
		if (reverse) {
			std::stable_sort(result->elements().rbegin(), result->elements().rend(), cmp);
		} else {
			std::stable_sort(result->elements().begin(), result->elements().end(), cmp);
		}

		if (err.is_err()) { return err; }
	} else {
		PyResult<PyObject *> err = Ok(py_none());
		std::vector<size_t> indices(cmp_list->elements().size());
		std::iota(indices.begin(), indices.end(), 0);
		// FIXME: should throw exception when comparing, as returning true is
		// probably messing up the C++ Compare requirment
		auto cmp = [&err, cmp_list](size_t lhs_index, size_t rhs_index) -> bool {
			const auto &lhs = cmp_list->elements()[lhs_index];
			const auto &rhs = cmp_list->elements()[rhs_index];
			if (auto cmp = less_than(lhs, rhs, VirtualMachine::the().interpreter()); cmp.is_ok()) {
				auto is_true = truthy(cmp.unwrap(), VirtualMachine::the().interpreter());
				if (is_true.is_err()) {
					err = Err(is_true.unwrap_err());
					return true;
				}
				return is_true.unwrap();
			} else {
				return false;
			}
		};
		if (reverse) {
			std::stable_sort(indices.rbegin(), indices.rend(), cmp);
		} else {
			std::stable_sort(indices.begin(), indices.end(), cmp);
		}

		if (err.is_err()) { return err; }

		cmp_list->elements().clear();
		for (const auto &index : indices) {
			cmp_list->elements().push_back(result->elements()[index]);
		}
		result = cmp_list;
	}

	return Ok(result);
}

PyResult<PyObject *> vars(PyTuple *args, PyDict *kwargs, Interpreter &interpreter)
{
	auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
		kwargs,
		"vars",
		std::integral_constant<size_t, 0>{},
		std::integral_constant<size_t, 1>{},
		nullptr);

	if (result.is_err()) { return Err(result.unwrap_err()); }
	auto [obj] = result.unwrap();

	if (!obj) { return Ok(interpreter.execution_frame()->locals()); }

	auto dict = obj->lookup_attribute(PyString::create("__dict__").unwrap());
	if (std::get<1>(dict) == LookupAttrResult::NOT_FOUND) {
		return Err(type_error("vars() argument must have __dict__ attribute"));
	}

	return std::get<0>(dict);
}

PyResult<PyObject *> divmod(PyTuple *args, PyDict *kwargs, Interpreter &)
{
	auto result = PyArgsParser<PyObject *, PyObject *>::unpack_tuple(args,
		kwargs,
		"divmod",
		std::integral_constant<size_t, 2>{},
		std::integral_constant<size_t, 2>{});

	if (result.is_err()) { return Err(result.unwrap_err()); }
	auto [lhs, rhs] = result.unwrap();

	return lhs->divmod(rhs);
}

PyResult<PyObject *> round(PyTuple *args, PyDict *kwargs, Interpreter &)
{
	auto result = PyArgsParser<PyObject *, PyObject *>::unpack_tuple(args,
		kwargs,
		"round",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 2>{},
		nullptr);

	if (result.is_err()) { return Err(result.unwrap_err()); }
	auto [value, ndigits] = result.unwrap();

	auto round = value->lookup_attribute(PyString::create("__round__").unwrap());

	if (std::get<1>(round) == LookupAttrResult::NOT_FOUND) {
		return Err(type_error("type {} doesn't define __round__ method", value->type()->name()));
	}
	if (std::get<0>(round).is_err()) { return std::get<0>(round); }

	if (!ndigits) { ndigits = py_none(); }
	return std::get<0>(round).unwrap()->call(PyTuple::create(ndigits).unwrap(), nullptr);
}

auto builtin_types()
{
	return std::array{
		types::type(),
		types::super(),
		types::bool_(),
		types::bytes(),
		types::bytearray(),
		types::ellipsis(),
		types::str(),
		types::float_(),
		types::integer(),
		types::complex(),
		types::none(),
		types::object(),
		types::memoryview(),
		types::dict(),
		types::list(),
		types::tuple(),
		types::range(),
		types::set(),
		types::frozenset(),
		types::property(),
		types::static_method(),
		types::classmethod(),
		types::slice(),
		types::reversed(),
		types::zip(),
		types::enumerate(),
		types::not_implemented(),
		types::map(),

	};
}

auto builtin_exceptions()
{
	return std::array{
		types::base_exception(),
		types::exception(),
		types::type_error(),
		types::assertion_error(),
		types::attribute_error(),
		types::value_error(),
		types::name_error(),
		types::runtime_error(),
		types::import_error(),
		types::key_error(),
		types::not_implemented_error(),
		types::module_not_found_error(),
		types::os_error(),
		types::lookup_error(),
		types::index_error(),
		types::warning(),
		types::deprecation_warning(),
		types::import_warning(),
		types::pending_deprecation_warning(),
		types::resource_warning(),
		types::syntax_error(),
		types::memory_error(),
		types::stop_iteration(),
		types::unbound_local_error(),
	};
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

	auto types = builtin_types();
	auto exceptions = builtin_exceptions();

	s_builtin_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("__builtins__").unwrap(),
		PyString::create("").unwrap())
						   .unwrap();

	for (auto *type : types) {
		s_builtin_module->add_symbol(PyString::create(type->name()).unwrap(), type);
	}

	for (auto *type : exceptions) {
		s_builtin_module->add_symbol(PyString::create(type->name()).unwrap(), type);
	}

	s_builtin_module->add_symbol(PyString::create("__build_class__").unwrap(),
		heap.allocate<PyNativeFunction>(
			"__build_class__", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return build_class(args, kwargs, interpreter);
			}));

	s_builtin_module->add_symbol(PyString::create("__import__").unwrap(),
		heap.allocate<PyNativeFunction>(
			"__import__", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return import(args, kwargs, interpreter);
			}));

	s_builtin_module->add_symbol(PyString::create("abs").unwrap(),
		heap.allocate<PyNativeFunction>("abs", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return abs(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("all").unwrap(),
		heap.allocate<PyNativeFunction>("all", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return all(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("any").unwrap(),
		heap.allocate<PyNativeFunction>("any", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return any(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("dir").unwrap(),
		heap.allocate<PyNativeFunction>("dir", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return dir(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("getattr").unwrap(),
		heap.allocate<PyNativeFunction>("getattr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return getattr(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("globals").unwrap(),
		heap.allocate<PyNativeFunction>("globals", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return globals(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("hasattr").unwrap(),
		heap.allocate<PyNativeFunction>("hasattr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return hasattr(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("hash").unwrap(),
		heap.allocate<PyNativeFunction>("hash", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return hash(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("hex").unwrap(),
		heap.allocate<PyNativeFunction>("hex", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return hex(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("id").unwrap(),
		heap.allocate<PyNativeFunction>("id", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return id(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("iter").unwrap(),
		heap.allocate<PyNativeFunction>("iter", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return iter(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("isinstance").unwrap(),
		heap.allocate<PyNativeFunction>(
			"isinstance", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return isinstance(args, kwargs, interpreter);
			}));

	s_builtin_module->add_symbol(PyString::create("issubclass").unwrap(),
		heap.allocate<PyNativeFunction>(
			"issubclass", [&interpreter](PyTuple *args, PyDict *kwargs) {
				return issubclass(args, kwargs, interpreter);
			}));

	s_builtin_module->add_symbol(PyString::create("locals").unwrap(),
		heap.allocate<PyNativeFunction>("locals", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return locals(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("len").unwrap(),
		heap.allocate<PyNativeFunction>("len", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return len(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("next").unwrap(),
		heap.allocate<PyNativeFunction>("next", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return next(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("ord").unwrap(),
		heap.allocate<PyNativeFunction>("ord", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return ord(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("chr").unwrap(),
		heap.allocate<PyNativeFunction>("chr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return chr(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("print").unwrap(),
		heap.allocate<PyNativeFunction>("print", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return print(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("repr").unwrap(),
		heap.allocate<PyNativeFunction>("repr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return repr(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("setattr").unwrap(),
		heap.allocate<PyNativeFunction>("setattr", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return setattr(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("exec").unwrap(),
		heap.allocate<PyNativeFunction>("exec", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return exec(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("callable").unwrap(),
		heap.allocate<PyNativeFunction>("callable", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return callable(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("compile").unwrap(),
		heap.allocate<PyNativeFunction>("compile", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return compile(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("max").unwrap(),
		heap.allocate<PyNativeFunction>("max", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return max(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("min").unwrap(),
		heap.allocate<PyNativeFunction>("min", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return min(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("eval").unwrap(),
		heap.allocate<PyNativeFunction>("eval", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return eval(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("ascii").unwrap(),
		heap.allocate<PyNativeFunction>("ascii", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return ascii(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("sorted").unwrap(),
		heap.allocate<PyNativeFunction>("sorted", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return sorted(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("vars").unwrap(),
		heap.allocate<PyNativeFunction>("vars", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return vars(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("divmod").unwrap(),
		heap.allocate<PyNativeFunction>("divmod", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return divmod(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("round").unwrap(),
		heap.allocate<PyNativeFunction>("round", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return round(args, kwargs, interpreter);
		}));

	return s_builtin_module;
}

}// namespace py
