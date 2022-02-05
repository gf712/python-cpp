#pragma once

#include "ExecutionFrame.hpp"

#include "forward.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/forward.hpp"
#include "vm/VM.hpp"

#include <string>
#include <string_view>

class BytecodeProgram;

class Interpreter
	: NonCopyable
	, NonMoveable
{
  public:
	enum class Status { OK, EXCEPTION };

  private:
	ExecutionFrame *m_current_frame{ nullptr };
	ExecutionFrame *m_global_frame{ nullptr };
	std::vector<py::PyModule *> m_available_modules;
	py::PyModule *m_module;
	Status m_status{ Status::OK };
	std::string m_entry_script;
	std::vector<std::string> m_argv;
	const Program *m_program;

  public:
	Interpreter();

	void set_status(Status status) { m_status = status; }
	Status status() const { return m_status; }

	template<typename... Ts> void raise_exception(py::PyObject *exception)
	{
		m_status = Status::EXCEPTION;
		m_current_frame->set_exception(std::move(exception));
	}

	ExecutionFrame *execution_frame() const { return m_current_frame; }
	ExecutionFrame *global_execution_frame() const { return m_global_frame; }

	void set_execution_frame(ExecutionFrame *frame) { m_current_frame = frame; }

	void store_object(const std::string &name, const py::Value &value)
	{
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
		m_current_frame->put_local(name, value);
		if (m_current_frame == m_global_frame) { m_current_frame->put_global(name, value); }
	}

	std::optional<py::Value> get_object(const std::string &name);

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
		const std::vector<std::string> &argnames,
		const std::vector<py::Value> &default_values,
		const std::vector<py::Value> &kw_default_values,
		size_t positional_args_count,
		size_t kwonly_args_count,
		const CodeFlags &flags) const;

	py::PyObject *call(const std::shared_ptr<Function> &, ExecutionFrame *function_frame);

	py::PyObject *call(py::PyNativeFunction *native_func, py::PyTuple *args, py::PyDict *kwargs);

  private:
	void internal_setup(const std::string &name,
		std::string entry_script,
		std::vector<std::string> argv,
		size_t local_registers);
};
