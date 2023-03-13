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
#include "runtime/warnings/ImportWarning.hpp"
#include "runtime/warnings/Warning.hpp"

#include "executable/Mangler.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/Bytecode.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"

#include "interpreter/Interpreter.hpp"

#include "lexer/Lexer.hpp"

#include "memory/GarbageCollector.hpp"

#include "parser/Parser.hpp"

#include "vm/VM.hpp"

#include "utilities.hpp"

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
	auto reprfunc = [](const PyResult<PyObject *> &arg) -> PyResult<PyObject *> {
		if (arg.is_err()) return arg;
		return arg.unwrap()->repr();
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
		auto reprobj_ = reprfunc(*arg_it);
		if (reprobj_.is_err()) { return reprobj_; }
		auto reprobj = reprobj_.unwrap();
		spdlog::debug("repr result: {}", reprobj->to_string());
		std::cout << reprobj->to_string() << separator;
		std::advance(arg_it, 1);
	}

	spdlog::debug("arg function ptr: {}", static_cast<void *>((*arg_it).unwrap()));
	auto reprobj_ = reprfunc(*arg_it);
	if (reprobj_.is_err()) { return reprobj_; }
	auto reprobj = reprobj_.unwrap();
	spdlog::debug("repr result: {}", reprobj->to_string());
	std::cout << reprobj->to_string() << end;

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
	// FIXME
	if (kwargs && kwargs->map().size() != 1) { TODO(); }
	auto metaclass_ = [kwargs]() -> PyResult<PyObject *> {
		if (kwargs && kwargs->map().size() == 1) {
			auto it = kwargs->map().find(String{ "metaclass" });
			ASSERT(it != kwargs->map().end());
			return PyObject::from(it->second);
		} else {
			return Ok(type());
		}
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

	auto ns_ = [metaclass, class_name, bases, kwargs]() -> PyResult<PyObject *> {
		if (metaclass == type()) {
			return PyDict::create();
		} else {
			auto prepare = PyString::create("__prepare__");
			if (prepare.is_err()) { return prepare; }
			auto prepare_kwargs = kwargs->map();
			prepare_kwargs.erase(String{ "metaclass" });
			auto new_kwargs_ = PyDict::create(prepare_kwargs);
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

	// this calls a function that defines a call
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
	return Ok(static_cast<PyObject *>(interpreter.execution_frame()->globals()));
}


PyResult<PyObject *> locals(const PyTuple *, const PyDict *, Interpreter &interpreter)
{
	return Ok(static_cast<PyObject *>(interpreter.execution_frame()->locals()));
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
	return PyInteger::create(static_cast<int64_t>(bit_cast<intptr_t>(obj.unwrap())));
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
		as<PyInteger>(level)->as_size_t());
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
#
	const auto &iterator = iterable->iter();
	if (iterator.is_err()) return iterator;
	auto next_value = iterator.unwrap()->next();
	while (!next_value.is_err()) {
		const auto is_truthy = next_value.unwrap()->true_();
		if (is_truthy.is_err()) return Err(is_truthy.unwrap_err());
		if (!is_truthy.unwrap()) { return Ok(py_false()); }
		next_value = iterator.unwrap()->next();
	}

	// FIXME: store StopIteration type somewhere so we don't have to instantiate a StopIteration
	//        exception object just to get its type
	if (next_value.unwrap_err()->type() == stop_iteration()->type()) {
		return Ok(py_true());
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

		std::shared_ptr<Program> bytecode = codegen::BytecodeGenerator::compile(
			p.module(), { filename_str }, compiler::OptimizationLevel::None);
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

auto builtin_types()
{
	return std::array{
		type(),
		super(),
		bool_(),
		bytes(),
		bytearray(),
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
		set(),
		frozenset(),
		property(),
		static_method(),
		classmethod(),
		slice(),
		reversed(),
		zip(),
		enumerate(),
	};
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
	ImportError::register_type(blt);
	KeyError::register_type(blt);
	NotImplementedError::register_type(blt);
	ModuleNotFoundError::register_type(blt);
	OSError::register_type(blt);
	LookupError::register_type(blt);
	IndexError::register_type(blt);
	Warning::register_type(blt);
	ImportWarning::register_type(blt);
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

	s_builtin_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("__builtins__").unwrap(),
		PyString::create("").unwrap())
						   .unwrap();

	for (auto *type : types) {
		s_builtin_module->add_symbol(PyString::create(type->name()).unwrap(), type);
	}

	initialize_exceptions(s_builtin_module);

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

	s_builtin_module->add_symbol(PyString::create("compile").unwrap(),
		heap.allocate<PyNativeFunction>("compile", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return compile(args, kwargs, interpreter);
		}));

	s_builtin_module->add_symbol(PyString::create("max").unwrap(),
		heap.allocate<PyNativeFunction>("max", [&interpreter](PyTuple *args, PyDict *kwargs) {
			return max(args, kwargs, interpreter);
		}));

	return s_builtin_module;
}

}// namespace py
