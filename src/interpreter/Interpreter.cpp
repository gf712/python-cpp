#include "Interpreter.hpp"

#include "runtime/BaseException.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"
#include "runtime/Value.hpp"
#include "runtime/modules/Modules.hpp"

#include "executable/Program.hpp"
#include "executable/bytecode/Bytecode.hpp"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

static PyString *s_main__ = nullptr;
static PyString *s_sys__ = nullptr;


Interpreter::Interpreter() {}

void Interpreter::internal_setup(const std::string &name,
	std::string entry_script,
	std::vector<std::string> argv,
	size_t local_registers)
{
	m_entry_script = std::move(entry_script);
	m_argv = std::move(argv);

	auto &heap = VirtualMachine::the().heap();

	// initialize the standard types by initializing the builtins module
	auto *builtins = builtins_module(*this);

	// before we construct the first PyObject we need to initialize the standard types
	auto name_ = PyString::create(name);
	auto *main_module = heap.allocate<PyModule>(name_);
	m_module = main_module;
	m_available_modules.push_back(main_module);
	m_available_modules.push_back(builtins);
	m_available_modules.push_back(sys_module(*this));

	if (!s_sys__) { s_sys__ = heap.allocate<PyString>("sys"); }
	PyDict::MapType global_map = { { String{ "__name__" }, name_ },
		{ String{ "__doc__" }, py_none() },
		{ String{ "__package__" }, py_none() } };

	for (auto *module : m_available_modules) { global_map[module->name()] = module; }

	auto *globals = VirtualMachine::the().heap().allocate<PyDict>(global_map);
	auto *locals = globals;
	m_current_frame = ExecutionFrame::create(nullptr, local_registers, globals, locals, nullptr);
	m_global_frame = m_current_frame;
}

void Interpreter::setup(std::shared_ptr<Program> program)
{
	const auto name = fs::path(program->filename()).stem();
	internal_setup(name, program->filename(), program->argv(), program->main_stack_size());
	m_program = std::move(program);
}

void Interpreter::setup_main_interpreter(std::shared_ptr<Program> program)
{
	auto &heap = VirtualMachine::the().heap();

	internal_setup("__main__", program->filename(), program->argv(), program->main_stack_size());
	if (!s_main__) { s_main__ = heap.allocate<PyString>("__main__"); }
	m_program = std::move(program);
}

void Interpreter::unwind()
{
	auto raised_exception = m_current_frame->exception();
	while (!m_current_frame->catch_exception(raised_exception)) {
		// don't unwind beyond the main frame
		if (!m_current_frame->parent()) {
			// uncaught exception
			ASSERT(m_current_frame->exception())
			std::cout << static_cast<BaseException *>(m_current_frame->exception())->what() << '\n';
			break;
		}
		m_current_frame = m_current_frame->exit();
	}
	m_current_frame->set_exception(nullptr);
}

PyModule *Interpreter::get_imported_module(PyString *name) const
{
	for (auto *module : m_available_modules) {
		if (module->name()->value() == name->value()) { return module; }
	}
	return nullptr;
}

const std::shared_ptr<Function> &Interpreter::functions(size_t idx) const
{
	return m_program->function(idx);
}

PyObject *Interpreter::call(const std::shared_ptr<Function> &func, ExecutionFrame *function_frame)
{
	auto &vm = VirtualMachine::the();
	function_frame->m_parent = m_current_frame;
	m_current_frame = function_frame;
	vm.call(func, function_frame->m_register_count);

	// cleanup
	auto current_frame = std::unique_ptr<ExecutionFrame, void (*)(ExecutionFrame *)>(
		m_current_frame, [](ExecutionFrame *ptr) {
			spdlog::debug("Deallocationg ExecutionFrame {}", (void *)ptr);
			ptr->~ExecutionFrame();
		});
	m_current_frame = current_frame->exit();

	return PyObject::from(vm.reg(0));
}

PyObject *Interpreter::call(PyNativeFunction *native_func, PyTuple *args, PyDict *kwargs)
{
	auto &vm = VirtualMachine::the();
	auto result = native_func->operator()(args, kwargs);

	spdlog::debug("Native function return value: {}", result->to_string());

	vm.reg(0) = result;
	return result;
}

std::optional<Value> Interpreter::get_object(const std::string &name)
{
	Value obj;

	const auto &name_value = String{ name };

	ASSERT(execution_frame()->locals())
	ASSERT(execution_frame()->globals())
	ASSERT(execution_frame()->builtins())

	const auto &locals = execution_frame()->locals()->map();
	const auto &globals = execution_frame()->globals()->map();
	const auto &builtins = execution_frame()->builtins()->symbol_table();

	if (auto it = locals.find(name_value); it != locals.end()) {
		obj = it->second;
	} else if (auto it = globals.find(name_value); it != globals.end()) {
		obj = it->second;
	} else {
		auto pystr_name = PyString::create(name);
		if (auto it = builtins.find(pystr_name); it != builtins.end()) {
			obj = it->second;
		} else {
			raise_exception("NameError: name '{:s}' is not defined", name);
			obj = py_none();
		}
	}

	return obj;
}
