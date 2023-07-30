#include "Import.hpp"
#include "ImportError.hpp"
#include "PyBool.hpp"
#include "PyCode.hpp"
#include "PyDict.hpp"
#include "PyFrame.hpp"
#include "PyModule.hpp"
#include "PyNone.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "ValueError.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "frozen/importlib.h"
#include "frozen/importlib_external.h"
#include "interpreter/Interpreter.hpp"
#include "modules/config.hpp"

namespace py {

namespace {
	// adapted from CPython import.c resolve_name
	PyResult<PyString *> resolve_name(PyString *name, PyDict *globals, uint32_t level)
	{
		if (!globals) { return Err(value_error("'__name__' not in globals")); }

		if (!globals->map().contains(String{ "__package__" })) {
			return Err(value_error("'__package__' not in globals"));
		}
		if (!globals->map().contains(String{ "__spec__" })) {
			return Err(value_error("'__spec__' not in globals"));
		}
		auto package_ = PyObject::from(globals->map().at(String{ "__package__" }));
		ASSERT(package_.is_ok())
		auto *package = package_.unwrap();
		auto spec_ = PyObject::from(globals->map().at(String{ "__spec__" }));
		ASSERT(spec_.is_ok())
		auto *spec = spec_.unwrap();

		if (package != py_none()) {
			if (!as<PyString>(package)) {
				return Err(type_error("package must be a string"));
			} else if (spec != py_none()) {
				auto parent = spec->get_attribute(PyString::create("parent").unwrap());
				if (parent.is_err()) return Err(parent.unwrap_err());
				auto are_equal = package->eq(parent.unwrap());
				if (are_equal.is_err()) return Err(are_equal.unwrap_err());
				if (are_equal.unwrap() == py_false()) {
					// TODO: emit import warning
					//       "__package__ != __spec__.parent"
				}
			}
		} else if (spec != py_none()) {
			package_ = spec->get_attribute(PyString::create("parent").unwrap());
			if (package_.is_err()) return Err(package_.unwrap_err());
			if (!as<PyString>(package_.unwrap())) {
				return Err(type_error("__spec__.parent must be a string"));
			}
			package = package_.unwrap();
		} else {
			// TODO: emit import warning
			// "can't resolve package from __spec__ or __package__, "
			// "falling back on __name__ and __path__"
			if (!globals->map().contains(String{ "__name__" })) {
				return Err(value_error("'__name__' not in globals"));
			}
			package_ = PyObject::from(globals->map().at(String{ "__name__" }));
			if (package_.is_err()) return Err(package_.unwrap_err());
			if (!as<PyString>(package)) { return Err(type_error("__name__ must be a string")); }
			package = package_.unwrap();

			const bool haspath = globals->map().contains(String{ "__path__" });

			if (!haspath) {
				// Unicode what?
				const auto &package_str = as<PyString>(package)->value();
				auto pos = package_str.find_last_of('.');
				if (pos == std::string::npos) {
					return Err(
						import_error("attempted relative import with no known parent package"));
				}
				package_ = PyString::create(package_str.substr(pos));
				if (package_.is_err()) return Err(package_.unwrap_err());
				package = package_.unwrap();
			}
		}

		ASSERT(as<PyString>(package))

		size_t last_dot = as<PyString>(package)->size();

		if (last_dot == 0) {
			return Err(import_error("attempted relative import with no known parent package"));
		}

		const auto &package_str = as<PyString>(package)->value();

		for (size_t level_up = 1; level_up < level; level_up++) {
			const auto pos = package_str.find_last_of('.', last_dot);
			if (pos == std::string::npos) {
				return Err(import_error("attempted relative import beyond top-level package"));
			}
			last_dot = pos;
		}

		auto base_ = PyString::create(package_str.substr(0, last_dot));
		if (base_.is_err()) return base_;
		auto *base = base_.unwrap();
		if (base->size() == 0) { return Ok(base); }

		auto abs_name_str = fmt::format("{}.{}", base->to_string(), name->to_string());
		auto abs_name = PyString::create(abs_name_str);

		return abs_name;
	}

	// adapted from CPython import.c import_get_module
	std::optional<PyModule *> import_get_module(PyString *name)
	{
		auto *available_modules = VirtualMachine::the().interpreter().modules();
		if (auto it = available_modules->map().find(name); it != available_modules->map().end()) {
			return as<PyModule>(PyObject::from(it->second).unwrap());
		}
		return {};
	}

	PyResult<PyModule *> import_add_module(PyString *name)
	{
		auto m = import_get_module(name);
		if (m.has_value()) return Ok(*m);
		return PyModule::create(PyDict::create().unwrap(), name, PyString::create("").unwrap());
	}

	void remove_module(PyString *name)
	{
		auto *available_modules = VirtualMachine::the().interpreter().modules();
		available_modules->remove(name);
	}

	PyResult<std::monostate> import_ensure_initialized(PyModule *module, PyString *name)
	{
		auto spec_str = PyString::create("__spec__").unwrap();
		if (module->symbol_table()->map().contains(spec_str)) {
			auto spec = PyObject::from(module->symbol_table()->map().at(spec_str));
			if (spec.is_err()) return Err(spec.unwrap_err());

			auto value = spec.unwrap()->get_attribute(PyString::create("_initializing").unwrap());
			if (value.is_err()) return Ok(std::monostate{});

			auto initializing = truthy(value.unwrap(), VirtualMachine::the().interpreter());
			if (initializing.is_err()) return Err(initializing.unwrap_err());
			if (initializing.unwrap()) {
				auto _lock_unlock_module = PyString::create("_lock_unlock_module");
				if (_lock_unlock_module.is_err()) return Err(_lock_unlock_module.unwrap_err());
				auto args = PyTuple::create(_lock_unlock_module.unwrap(), name);
				if (args.is_err()) return Err(args.unwrap_err());
				value =
					VirtualMachine::the().interpreter().importlib()->call(args.unwrap(), nullptr);
				if (value.is_err()) return Err(value.unwrap_err());
				return Ok(std::monostate{});
			}
		}
		return Ok(std::monostate{});
	}

	PyResult<PyModule *> import_find_and_load(PyString *name)
	{
		auto find_and_load = PyString::create("_find_and_load");
		if (find_and_load.is_err()) return Err(find_and_load.unwrap_err());
		auto import_find_and_load =
			VirtualMachine::the().interpreter().importlib()->get_method(find_and_load.unwrap());
		if (import_find_and_load.is_err()) return Err(import_find_and_load.unwrap_err());

		auto args = PyTuple::create(name, VirtualMachine::the().interpreter().importfunc());
		if (args.is_err()) return Err(args.unwrap_err());

		auto module = import_find_and_load.unwrap()->call(args.unwrap(), nullptr);
		if (module.is_err()) return Err(module.unwrap_err());
		if (!as<PyModule>(module.unwrap())) {
			return Err(type_error("expected module to be of type module"));
		}
		return Ok(as<PyModule>(module.unwrap()));
	}

	std::array frozen_modules = {
		FrozenModule{ .name = "_frozen_importlib", .code = _bootstrap, .is_package = false },
		FrozenModule{ .name = "_frozen_importlib_external",
			.code = _bootstrap_external,
			.is_package = false },
	};

}// namespace

std::optional<std::reference_wrapper<const FrozenModule>> find_frozen(PyString *name)
{
	for (const auto &m : frozen_modules) {
		if (m.name == name->value()) { return m; }
	}
	return {};
}


// adapted from CPython import.c PyImport_ImportModuleLevelObject
PyResult<PyObject *> import_module_level_object(PyString *name,
	PyDict *globals,
	PyObject *locals,
	PyObject *fromlist,
	uint32_t level)
{
	(void)locals;
	(void)fromlist;
	if (!name) { return Err(value_error("Empty module name")); }

	auto absolute_name = [&]() -> PyResult<PyString *> {
		if (level > 0) {
			return resolve_name(name, globals, level);
		} else {
			if (name->size() == 0) { return Err(value_error("Empty module name")); }
			return Ok(name);
		}
	}();

	if (absolute_name.is_err()) return absolute_name;

	auto module = [&]() -> PyResult<PyModule *> {
		auto module = import_get_module(absolute_name.unwrap());
		if (module.has_value()) {
			auto is_initialized = import_ensure_initialized(*module, absolute_name.unwrap());
			if (is_initialized.is_err()) {
				return Err(is_initialized.unwrap_err());
			} else {
				return Ok(*module);
			}
		} else {
			return import_find_and_load(absolute_name.unwrap());
		}
	}();

	return module;
}

// adapted from CPython import.c PyImport_ImportFrozenModuleObject
PyResult<PyModule *> import_frozen_module(PyString *name)
{
	const auto frozen_module = find_frozen(name);
	if (!frozen_module.has_value()) { return Err(import_error("")); }

	const auto &serialised_code = frozen_module->get().code;
	std::shared_ptr<Program> program = BytecodeProgram::deserialize(serialised_code);

	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();

	if (!program) { return Err(import_error("TODO")); }

	if (frozen_module->get().is_package) { TODO(); }

	auto module = import_add_module(name);
	if (module.is_err()) return module;
	module.unwrap()->set_program(program);

	auto &vm = VirtualMachine::the();

	module.unwrap()->symbol_table()->insert(String{ "__builtins__" }, vm.interpreter().builtins());

	auto result = [&vm, &program, module]() {
		auto *code = as<PyCode>(static_cast<BytecodeProgram &>(*program).main_function());
		ASSERT(code)
		auto *function_frame =
			PyFrame::create(VirtualMachine::the().interpreter().execution_frame(),
				code->register_count(),
				0,
				code,
				module.unwrap()->symbol_table(),
				module.unwrap()->symbol_table(),
				code->consts(),
				code->names(),
				nullptr);
		[[maybe_unused]] auto scoped_stack =
			vm.interpreter().setup_call_stack(code->function(), function_frame);
		return vm.interpreter().call(code->function(), function_frame);
	}();

	if (result.is_err()) {
		remove_module(name);
		spdlog::error("{}", result.unwrap_err()->to_string());
		TODO();
		return Err(import_error("TODO"));
	}

	module.unwrap()->set_program(std::move(program));

	return module;
}

// adapted from _imp_create_builtin(PyObject *module, PyObject *spec)
PyResult<PyObject *> create_builtin(PyObject *spec)
{
	PyObject *mod{ nullptr };

	auto name = spec->get_attribute(PyString::create("name").unwrap());
	if (name.is_err()) return name;

	auto namestr = as<PyString>(name.unwrap());
	if (!namestr) {
		return Err(type_error(
			"expected spec name to str, but got {}", name.unwrap()->type()->to_string()));
	}

	for (const auto &[name, init_func] : builtin_modules) {
		if (name == namestr->value()) {
			if (!init_func) { return import_add_module(namestr); }
			mod = init_func();
			// TODO: init_func should return PyResult<PyModule*>
			if (!mod) { TODO(); }
			// FIXME: this isn't quite right, see cpython _imp_create_builtin
			return Ok(mod);
		}
	}

	return Ok(py_none());
}

}// namespace py