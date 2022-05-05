#include "Interpreter.hpp"

#include "runtime/BaseException.hpp"
#include "runtime/NameError.hpp"
#include "runtime/PyCode.hpp"
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
#include "executable/bytecode/BytecodeProgram.hpp"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace py;

static PyString *s_main__ = nullptr;
static PyString *s_sys__ = nullptr;


Interpreter::Interpreter() {}

void Interpreter::internal_setup(const std::string &name,
	std::string entry_script,
	std::vector<std::string> argv,
	size_t local_registers,
	const PyTuple *consts)
{
	m_entry_script = std::move(entry_script);
	m_argv = std::move(argv);

	auto &heap = VirtualMachine::the().heap();

	// initialize the standard types by initializing the builtins module
	auto *builtins = builtins_module(*this);

	auto name_ = PyString::create(name);
	if (name_.is_err()) { TODO(); }
	auto *main_module = heap.allocate<PyModule>(name_.unwrap());
	if (!main_module) { TODO(); }
	m_module = main_module;
	m_available_modules.push_back(main_module);
	m_available_modules.push_back(builtins);
	m_available_modules.push_back(sys_module(*this));

	if (!s_sys__) { s_sys__ = heap.allocate<PyString>("sys"); }
	PyDict::MapType global_map = { { String{ "__name__" }, name_.unwrap() },
		{ String{ "__doc__" }, py_none() },
		{ String{ "__package__" }, py_none() } };

	for (auto *module : m_available_modules) { global_map[module->name()] = module; }

	auto *globals = VirtualMachine::the().heap().allocate<PyDict>(global_map);
	auto *locals = globals;
	m_current_frame = PyFrame::create(nullptr, local_registers, 0, globals, locals, consts);
	m_global_frame = m_current_frame;

	m_importlib = nullptr;
	m_import_func = nullptr;
}

void Interpreter::setup(const BytecodeProgram &program)
{
	m_program = static_cast<const Program *>(&program);

	const auto name = fs::path(program.filename()).stem();
	internal_setup(name,
		program.filename(),
		program.argv(),
		program.main_stack_size(),
		program.main_function()->consts());
}

void Interpreter::setup_main_interpreter(const BytecodeProgram &program)
{
	m_program = static_cast<const Program *>(&program);
	auto &heap = VirtualMachine::the().heap();

	internal_setup("__main__",
		program.filename(),
		program.argv(),
		program.main_stack_size(),
		program.main_function()->consts());
	if (!s_main__) { s_main__ = heap.allocate<PyString>("__main__"); }
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
	for (auto *module : m_available_modules) {
		if (module->name()->value() == name->value()) { return module; }
	}
	return nullptr;
}

PyObject *Interpreter::make_function(const std::string &function_name,
	const std::vector<py::Value> &default_values,
	const std::vector<py::Value> &kw_default_values,
	const std::vector<py::PyCell *> &closure) const
{
	auto *f = m_program->as_pyfunction(function_name, default_values, kw_default_values, closure);
	ASSERT(f)
	return f;
}

ScopedStack::~ScopedStack()
{
	auto &vm = VirtualMachine::the();
	if (&vm.stack().top() == &top_frame) { vm.pop_frame(); }
}

ScopedStack Interpreter::setup_call_stack(const std::unique_ptr<Function> &func,
	PyFrame *function_frame)
{
	auto &vm = VirtualMachine::the();
	vm.setup_call_stack(
		function_frame->m_register_count, func->stack_size() + function_frame->freevars().size());
	return ScopedStack{ vm.stack().top() };
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

PyResult<PyObject *> Interpreter::call(PyNativeFunction *native_func, PyTuple *args, PyDict *kwargs)
{
	auto &vm = VirtualMachine::the();
	auto result = native_func->operator()(args, kwargs);

	if (result.is_err()) { return result; }

	spdlog::debug("Native function return value: {}", result.unwrap()->to_string());

	vm.reg(0) = result.unwrap();
	return result;
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
	if (m_current_frame == m_global_frame) {
		m_current_frame->put_global(name, value);
	} else {
		m_current_frame->put_local(name, value);
	}
}

PyResult<Value> Interpreter::get_object(const std::string &name)
{
	ASSERT(execution_frame()->locals())
	ASSERT(execution_frame()->globals())
	ASSERT(execution_frame()->builtins())

	const auto &locals = execution_frame()->locals()->map();
	const auto &globals = execution_frame()->globals()->map();
	const auto &builtins = execution_frame()->builtins()->symbol_table();

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
