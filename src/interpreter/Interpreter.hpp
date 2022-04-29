#pragma once


#include "forward.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/forward.hpp"
#include "vm/VM.hpp"

#include <string>
#include <string_view>

class BytecodeProgram;

struct ScopedStack
{
	const StackFrame &top_frame;
	~ScopedStack();
};

class Interpreter
	: NonCopyable
	, NonMoveable
{
  private:
	py::PyFrame *m_current_frame{ nullptr };
	py::PyFrame *m_global_frame{ nullptr };
	std::vector<py::PyModule *> m_available_modules;
	py::PyModule *m_module;
	py::PyModule *m_importlib;
	py::PyObject *m_import_func;
	std::string m_entry_script;
	std::vector<std::string> m_argv;
	const Program *m_program;

  public:
	Interpreter();

	void raise_exception(py::BaseException *exception)
	{
		m_current_frame->push_exception(std::move(exception));
	}

	py::PyFrame *execution_frame() const { return m_current_frame; }
	py::PyFrame *global_execution_frame() const { return m_global_frame; }

	void set_execution_frame(py::PyFrame *frame) { m_current_frame = frame; }

	void store_object(const std::string &name, const py::Value &value);

	py::PyResult get_object(const std::string &name);

	template<typename PyObjectType, typename... Args>
	py::PyObject *allocate_object(const std::string &name, Args &&... args)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto obj = heap.allocate<PyObjectType>(std::forward<Args>(args)...)) {
			store_object(name, obj);
			return obj;
		} else {
			return nullptr;
		}
	}

	py::PyModule *get_imported_module(py::PyString *) const;
	const std::vector<py::PyModule *> &get_available_modules() const { return m_available_modules; }

	py::PyModule *module() const { return m_module; }

	void unwind();

	void setup(const BytecodeProgram &program);
	void setup_main_interpreter(const BytecodeProgram &program);

	const std::string &entry_script() const { return m_entry_script; }
	const std::vector<std::string> &argv() const { return m_argv; }

	py::PyObject *make_function(const std::string &function_name,
		const std::vector<py::Value> &default_values,
		const std::vector<py::Value> &kw_default_values,
		const std::vector<py::PyCell *> &closure) const;

	ScopedStack setup_call_stack(const std::unique_ptr<Function> &, py::PyFrame *function_frame);
	py::PyResult call(const std::unique_ptr<Function> &, py::PyFrame *function_frame);

	py::PyResult call(py::PyNativeFunction *native_func, py::PyTuple *args, py::PyDict *kwargs);

	const Program *program() const { return m_program; }

  private:
	void internal_setup(const std::string &name,
		std::string entry_script,
		std::vector<std::string> argv,
		size_t local_registers,
		const py::PyTuple *consts);
};
