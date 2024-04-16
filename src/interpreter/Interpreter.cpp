#include "Interpreter.hpp"

#include "runtime/BaseException.hpp"
#include "runtime/Import.hpp"
#include "runtime/KeyError.hpp"
#include "runtime/NameError.hpp"
#include "runtime/PyCode.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"
#include "runtime/Value.hpp"
#include "runtime/modules/Modules.hpp"
#include "runtime/modules/config.hpp"
#include "runtime/types/builtin.hpp"

#include "executable/Program.hpp"
#include "executable/bytecode/Bytecode.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"

#include <filesystem>

namespace fs = std::filesystem;
using namespace py;

void initialize_types()
{
	[[maybe_unused]] auto scope_static_alloc =
		VirtualMachine::the().heap().scoped_static_allocation();
	types::object();
	types::type();
	types::super();
	types::bool_();
	types::bytes();
	types::bytes_iterator();
	types::bytearray();
	types::bytearray_iterator();
	types::memoryview();
	types::ellipsis();
	types::str();
	types::str_iterator();
	types::float_();
	types::integer();
	types::none();
	types::module();
	types::dict();
	types::dict_items();
	types::dict_items_iterator();
	types::dict_keys();
	types::dict_key_iterator();
	types::dict_values();
	types::dict_value_iterator();
	types::list();
	types::list_iterator();
	types::list_reverseiterator();
	types::tuple();
	types::tuple_iterator();
	types::set();
	types::frozenset();
	types::set_iterator();
	types::range();
	types::range_iterator();
	types::reversed();
	types::zip();
	types::enumerate();
	types::slice();
	types::function();
	types::native_function();
	types::llvm_function();
	types::code();
	types::cell();
	types::builtin_method();
	types::slot_wrapper();
	types::bound_method();
	types::method_wrapper();
	types::classmethod_descriptor();
	types::getset_descriptor();
	types::static_method();
	types::property();
	types::classmethod();
	types::member_descriptor();
	types::traceback();
	types::not_implemented();
	types::frame();
	types::namespace_();
	types::generator();
	types::coroutine();
	types::async_generator();
	types::mappingproxy();
	types::map();

	types::base_exception();
	types::exception();
	types::type_error();
	types::assertion_error();
	types::attribute_error();
	types::value_error();
	types::name_error();
	types::runtime_error();
	types::import_error();
	types::key_error();
	types::not_implemented_error();
	types::module_not_found_error();
	types::os_error();
	types::lookup_error();
	types::index_error();
	types::warning();
	types::import_warning();
	types::syntax_error();
	types::memory_error();
	types::stop_iteration();
}

Interpreter::Interpreter() {}

void Interpreter::internal_setup(const std::string &name,
	std::string entry_script,
	std::vector<std::string> argv,
	size_t local_registers,
	const PyTuple *consts,
	const std::vector<std::string> &names,
	Config &&config,
	std::shared_ptr<Program> &&program)
{
	PyModule *sys = nullptr;
	{
		[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();

		initialize_types();
		m_modules = PyDict::create().unwrap();
		m_entry_script = std::move(entry_script);
		m_argv = std::move(argv);

		m_builtins = builtins_module(*this);
		m_modules->insert(String{ "builtins" }, m_builtins);
		sys = sys_module(*this);
		m_modules->insert(String{ "sys" }, sys);

		auto name_ = PyString::create(name);
		if (name_.is_err()) { TODO(); }
		auto *main_module = PyModule::create(
			PyDict::create().unwrap(), name_.unwrap(), PyString::create("").unwrap())
								.unwrap();
		if (!main_module) { TODO(); }
		m_module = main_module;
		m_module->set_program(std::move(program));
		m_module->add_symbol(PyString::create("__builtins__").unwrap(), m_builtins);
		m_modules->insert(name_.unwrap(), m_module);
	}

	auto code = PyCode::create(m_module->program());
	if (code.is_err()) { TODO(); }

	auto *globals = m_module->symbol_table();
	auto *locals = globals;
	m_current_frame = PyFrame::create(
		nullptr, local_registers, 0, code.unwrap(), globals, locals, consts, names, nullptr);
	m_global_frame = m_current_frame;

	for (const auto &[name, module_factory] : builtin_modules) {
		if (module_factory) { m_modules->insert(String{ std::string{ name } }, module_factory()); }
	}

	if (config.requires_importlib) {
		auto *_imp = imp_module();
		auto importlib_name = PyString::create("_frozen_importlib").unwrap();
		auto importlib = import_frozen_module(importlib_name);
		ASSERT(importlib.is_ok())
		m_importlib = importlib.unwrap();
		m_modules->insert(String{ "_frozen_importlib" }, m_importlib);

		m_import_func =
			PyObject::from(m_builtins->symbol_table()->map().at(String{ "__import__" })).unwrap();
		auto install = m_importlib->get_method(PyString::create("_install").unwrap());
		if (install.is_err()) { TODO(); }

		ASSERT(sys);
		auto args = PyTuple::create(sys, _imp);
		if (args.is_err()) { TODO(); }
		auto result = install.unwrap()->call(args.unwrap(), nullptr);
		if (result.is_err()) {
			spdlog::error("Error calling _install: {}", result.unwrap_err()->to_string());
			TODO();
		}

		auto install_external =
			m_importlib->get_method(PyString::create("_install_external_importers").unwrap());
		if (install_external.is_err()) { TODO(); }

		result = install_external.unwrap()->call(PyTuple::create().unwrap(), nullptr);
		if (result.is_err()) {
			spdlog::error(
				"Error calling _install_external_importers: {}", result.unwrap_err()->to_string());
			TODO();
		}
	}
}

void Interpreter::setup(std::shared_ptr<BytecodeProgram> &&program)
{
	const auto name = fs::path(program->filename()).stem();
	auto *code = as<PyCode>(program->main_function());
	ASSERT(code);
	const auto &filename = program->filename();
	const auto &argv = program->argv();
	const auto &main_stack_size = program->main_stack_size();
	internal_setup(name,
		filename,
		argv,
		main_stack_size,
		code->consts(),
		code->names(),
		Config{ .requires_importlib = false },
		std::move(program));
}

void Interpreter::setup_main_interpreter(std::shared_ptr<BytecodeProgram> &&program)
{
	auto *code = as<PyCode>(program->main_function());
	ASSERT(code);
	const auto &filename = program->filename();
	const auto &argv = program->argv();
	const auto &main_stack_size = program->main_stack_size();
	internal_setup("__main__",
		filename,
		argv,
		main_stack_size,
		code->consts(),
		code->names(),
		Config{ .requires_importlib = true },
		std::move(program));
}

void Interpreter::raise_exception(py::BaseException *exception)
{
	m_current_frame->push_exception(exception);
}

PyModule *Interpreter::get_imported_module(PyString *name) const
{
	if (auto it = m_modules->map().find(name); it != m_modules->map().end()) {
		return as<PyModule>(PyObject::from(it->second).unwrap());
	}
	return nullptr;
}

ScopedStack::~ScopedStack()
{
	auto &vm = VirtualMachine::the();
	if (!vm.stack().empty() && top_frame && &vm.stack().top().get() == top_frame.get()) {
		vm.pop_frame();
	}
}

std::unique_ptr<StackFrame> ScopedStack::release()
{
	ASSERT(top_frame);
	auto &vm = VirtualMachine::the();
	vm.pop_frame();
	return StackFrame::create(top_frame->clone());
}

ScopedStack Interpreter::setup_call_stack(const std::unique_ptr<Function> &func,
	PyFrame *function_frame)
{
	auto &vm = VirtualMachine::the();
	auto frame = vm.setup_call_stack(function_frame->m_register_count,
		func->locals_count(),
		func->stack_size() + function_frame->freevars().size());
	return ScopedStack{ std::move(frame) };
}

PyResult<PyObject *> Interpreter::call(const std::unique_ptr<Function> &func,
	PyFrame *function_frame)
{
	auto &vm = VirtualMachine::the();
	function_frame->m_f_back = m_current_frame;
	m_current_frame = function_frame;
	auto result = func->call(vm, *this);

	// cleanup: the current_frame will be garbage collected
	m_current_frame = m_current_frame->exit();

	return result.and_then([](const auto &value) { return PyObject::from(value); });
}

PyResult<PyObject *> Interpreter::call(const std::unique_ptr<Function> &func,
	PyFrame *function_frame,
	StackFrame &stack_frame)
{
	auto &vm = VirtualMachine::the();
	function_frame->m_f_back = m_current_frame;
	m_current_frame = function_frame;

	stack_frame.return_address = VirtualMachine::the().instruction_pointer();
	stack_frame.restore();
	auto result = func->call_without_setup(vm, *this);

	// cleanup: the current_frame will be garbage collected
	m_current_frame = m_current_frame->exit();

	return result.and_then([](const auto &value) { return PyObject::from(value); });
}

PyResult<PyObject *> Interpreter::call(PyNativeFunction *native_func, PyTuple *args, PyDict *kwargs)
{
	auto &vm = VirtualMachine::the();
	ASSERT(native_func->is_function());
	return native_func->operator()(args, kwargs).and_then([&vm](PyObject *result) {
		spdlog::debug("Native function return value: {}", result->to_string());
		vm.reg(0) = result;
		return Ok(result);
	});
}

PyResult<PyObject *>
	Interpreter::call(PyNativeFunction *native_func, PyObject *self, PyTuple *args, PyDict *kwargs)
{
	auto &vm = VirtualMachine::the();
	ASSERT(native_func->is_method());
	return native_func->operator()(self, args, kwargs).and_then([&vm](PyObject *result) {
		spdlog::debug("Native function return value: {}", result->to_string());
		vm.reg(0) = result;
		return Ok(result);
	});
}

PyResult<std::monostate> Interpreter::store_object(const std::string &name, const Value &value)
{
	if (spdlog::get_level() == spdlog::level::debug) {
		spdlog::debug("Interpreter::store_object(name={}, value={}, current_frame={})",
			name,
			std::visit(
				[](const auto &val) {
					std::ostringstream os;
					os << val;
					return os.str();
				},
				value),
			(void *)m_current_frame);
	}
	return m_current_frame->put_local(name, value);
}

PyResult<Value> Interpreter::get_object(const std::string &name)
{
	ASSERT(execution_frame()->locals())
	ASSERT(execution_frame()->globals())
	ASSERT(execution_frame()->builtins())

	auto *locals = execution_frame()->locals();
	auto *globals = execution_frame()->globals();
	const auto &builtins = execution_frame()->builtins()->symbol_table()->map();

	return [&]() -> PyResult<Value> {
		const auto &name_value = String{ name };
		PyString *pystr_name = nullptr;

		if (auto *locals_ = as<PyDict>(locals)) {
			if (const auto &it = locals_->map().find(name_value); it != locals_->map().end()) {
				return Ok(std::move(it->second));
			}
		} else {
			if (!pystr_name) {
				auto name_ = PyString::create(name);
				if (name_.is_err()) { return name_; }
				pystr_name = name_.unwrap();
			}
			if (auto r = locals->as_mapping().unwrap().getitem(pystr_name); r.is_ok()) {
				return r;
			} else if (r.unwrap_err()->type() != KeyError::class_type()) {
				return r;
			}
		}

		if (auto *globals_ = as<PyDict>(globals)) {
			if (const auto &it = globals_->map().find(name_value); it != globals_->map().end()) {
				return Ok(std::move(it->second));
			}
		} else {
			if (!pystr_name) {
				auto name_ = PyString::create(name);
				if (name_.is_err()) { return name_; }
				pystr_name = name_.unwrap();
			}
			if (auto r = globals->as_mapping().unwrap().getitem(pystr_name); r.is_ok()) {
				return r;
			} else if (r.unwrap_err()->type() != KeyError::class_type()) {
				return r;
			}
		}

		if (!pystr_name) {
			auto name_ = PyString::create(name);
			if (name_.is_err()) { return name_; }
			pystr_name = name_.unwrap();
		}
		if (const auto &it = builtins.find(pystr_name); it != builtins.end()) {
			return Ok(std::move(it->second));
		}
		return Err(name_error("name '{:s}' is not defined", name));
	}();
}

void Interpreter::visit_graph(::Cell::Visitor &visitor)
{
	if (m_current_frame) {
		visitor.visit(*m_current_frame);
		auto *frame = m_current_frame;
		while (frame) {
			frame->code()->program()->visit_functions(visitor);
			frame = frame->parent();
		}
	}
	if (m_global_frame) visitor.visit(*m_global_frame);
	if (m_modules) visitor.visit(*m_modules);
	if (m_module) visitor.visit(*m_module);
	if (m_builtins) visitor.visit(*m_builtins);
	if (m_importlib) visitor.visit(*m_importlib);
	if (m_import_func) visitor.visit(*m_import_func);
}
