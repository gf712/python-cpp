#pragma once

#include "forward.hpp"
#include "runtime/forward.hpp"
#include "vm/VM.hpp"

#include <string>
#include <string_view>

class BytecodeProgram;

struct ScopedStack
{
	std::unique_ptr<StackFrame> top_frame;
	~ScopedStack();

	std::unique_ptr<StackFrame> release();
};

class Interpreter
	: NonCopyable
	, NonMoveable
{
  private:
	py::PyFrame *m_current_frame{ nullptr };
	py::PyFrame *m_global_frame{ nullptr };
	py::PyDict *m_modules{ nullptr };
	py::PyModule *m_module{ nullptr };
	py::PyModule *m_builtins{ nullptr };
	py::PyModule *m_importlib{ nullptr };
	py::PyObject *m_import_func{ nullptr };
	std::string m_entry_script;
	std::vector<std::string> m_argv;

  public:
	struct Config
	{
		bool requires_importlib;
	};

  public:
	Interpreter();

	void raise_exception(py::BaseException *exception);

	py::PyFrame *execution_frame() const { return m_current_frame; }
	py::PyFrame *global_execution_frame() const { return m_global_frame; }

	void set_execution_frame(py::PyFrame *frame) { m_current_frame = frame; }

	void store_object(const std::string &name, const py::Value &value);

	py::PyResult<py::Value> get_object(const std::string &name);

	template<typename PyObjectType, typename... Args>
	py::PyObject *allocate_object(const std::string &name, Args &&...args)
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

	py::PyDict *modules() const { return m_modules; }

	py::PyModule *importlib() const { return m_importlib; }

	py::PyObject *importfunc() const { return m_import_func; }

	py::PyModule *builtins() const { return m_builtins; }

	py::PyModule *module() const { return m_module; }

	void unwind();

	void setup(std::shared_ptr<BytecodeProgram> &&program);
	void setup_main_interpreter(std::shared_ptr<BytecodeProgram> &&program);

	const std::string &entry_script() const { return m_entry_script; }
	const std::vector<std::string> &argv() const { return m_argv; }

	ScopedStack setup_call_stack(const std::unique_ptr<Function> &, py::PyFrame *function_frame);
	py::PyResult<py::PyObject *> call(const std::unique_ptr<Function> &,
		py::PyFrame *function_frame);

	py::PyResult<py::PyObject *>
		call(const std::unique_ptr<Function> &, py::PyFrame *function_frame, StackFrame &frame);

	py::PyResult<py::PyObject *>
		call(py::PyNativeFunction *native_func, py::PyTuple *args, py::PyDict *kwargs);

	py::PyResult<py::PyObject *> call(py::PyNativeFunction *native_func,
		py::PyObject *self,
		py::PyTuple *args,
		py::PyDict *kwargs);

	void visit_graph(::Cell::Visitor &);

  private:
	void internal_setup(const std::string &name,
		std::string entry_script,
		std::vector<std::string> argv,
		size_t local_registers,
		const py::PyTuple *consts,
		const std::vector<std::string> &names,
		Config &&,
		std::shared_ptr<Program> &&);
};

void initialize_types();