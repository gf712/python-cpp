#include "Interpreter.hpp"

#include "runtime/BaseException.hpp"
#include "runtime/Import.hpp"
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
	object();
	type();
	super();
	bool_();
	bytes();
	bytes_iterator();
	bytearray();
	bytearray_iterator();
	ellipsis();
	str();
	str_iterator();
	float_();
	integer();
	none();
	module();
	dict();
	dict_items();
	dict_items_iterator();
	dict_keys();
	dict_key_iterator();
	dict_values();
	dict_value_iterator();
	list();
	list_iterator();
	list_reverseiterator();
	tuple();
	tuple_iterator();
	set();
	frozenset();
	set_iterator();
	range();
	range_iterator();
	reversed();
	slice();
	function();
	native_function();
	llvm_function();
	code();
	cell();
	builtin_method();
	slot_wrapper();
	bound_method();
	method_wrapper();
	classmethod_descriptor();
	getset_descriptor();
	static_method();
	property();
	classmethod();
	member_descriptor();
	traceback();
	not_implemented();
	frame();
	namespace_();
	generator();
	coroutine();
	async_generator();
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
	initialize_types();
	m_modules = PyDict::create().unwrap();
	m_entry_script = std::move(entry_script);
	m_argv = std::move(argv);

	// initialize the standard types by initializing the builtins module
	m_builtins = builtins_module(*this);
	m_modules->insert(String{ "builtins" }, m_builtins);
	auto *sys = sys_module(*this);
	m_modules->insert(String{ "sys" }, sys);

	auto name_ = PyString::create(name);
	if (name_.is_err()) { TODO(); }
	auto *main_module =
		PyModule::create(PyDict::create().unwrap(), name_.unwrap(), PyString::create("").unwrap())
			.unwrap();
	if (!main_module) { TODO(); }
	m_module = main_module;
	main_module->set_program(std::move(program));
	main_module->add_symbol(PyString::create("__builtins__").unwrap(), m_builtins);
	m_modules->insert(name_.unwrap(), main_module);

	auto code = PyCode::create(main_module->program());
	if (code.is_err()) { TODO(); }

	auto *globals = main_module->symbol_table();
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
	ASSERT(code)
	internal_setup(name,
		program->filename(),
		program->argv(),
		program->main_stack_size(),
		code->consts(),
		code->names(),
		Config{ .requires_importlib = false },
		std::move(program));
}

void Interpreter::setup_main_interpreter(std::shared_ptr<BytecodeProgram> &&program)
{
	auto *code = as<PyCode>(program->main_function());
	ASSERT(code)
	internal_setup("__main__",
		program->filename(),
		program->argv(),
		program->main_stack_size(),
		code->consts(),
		code->names(),
		Config{ .requires_importlib = true },
		std::move(program));
}

void Interpreter::raise_exception(py::BaseException *exception)
{
	m_current_frame->push_exception(exception);
}


void Interpreter::unwind()
{
	ASSERT(m_current_frame->exception_info().has_value())
	auto *raised_exception = m_current_frame->exception_info()->exception;
	while (!m_current_frame->catch_exception(raised_exception)) {
		// don't unwind beyond the main frame
		if (!m_current_frame->parent()) {
			// uncaught exception
			std::cout << static_cast<const BaseException *>(raised_exception)->format_traceback()
					  << '\n';
			break;
		}
		m_current_frame = m_current_frame->exit();
	}
	m_current_frame->pop_exception();
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
	return std::move(top_frame);
}

ScopedStack Interpreter::setup_call_stack(const std::unique_ptr<Function> &func,
	PyFrame *function_frame)
{
	auto &vm = VirtualMachine::the();
	auto frame = vm.setup_call_stack(
		function_frame->m_register_count, func->stack_size() + function_frame->freevars().size());
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

void Interpreter::store_object(const std::string &name, const Value &value)
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
	m_current_frame->put_local(name, value);
}

PyResult<Value> Interpreter::get_object(const std::string &name)
{
	ASSERT(execution_frame()->locals())
	ASSERT(execution_frame()->globals())
	ASSERT(execution_frame()->builtins())

	const auto &locals = execution_frame()->locals()->map();
	const auto &globals = execution_frame()->globals()->map();
	const auto &builtins = execution_frame()->builtins()->symbol_table()->map();

	return [&]() -> PyResult<Value> {
		const auto &name_value = String{ name };

		if (const auto &it = locals.find(name_value); it != locals.end()) {
			return Ok(std::move(it->second));
		} else if (const auto &it = globals.find(name_value); it != globals.end()) {
			return Ok(std::move(it->second));
		} else {
			auto pystr_name = PyString::create(name);
			if (pystr_name.is_err()) { return Err(pystr_name.unwrap_err()); }
			if (const auto &it = builtins.find(pystr_name.unwrap()); it != builtins.end()) {
				return Ok(std::move(it->second));
			} else {
				return Err(name_error("name '{:s}' is not defined", name));
			}
		}
		return Ok(Value{ py_none() });
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
