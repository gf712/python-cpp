#pragma once

#include "ExecutionFrame.hpp"

#include "forward.hpp"
#include "runtime/forward.hpp"
#include "vm/VM.hpp"

#include <string>
#include <string_view>

class Interpreter
	: NonCopyable
	, NonMoveable
{
  public:
	enum class Status { OK, EXCEPTION };

  private:
	ExecutionFrame *m_current_frame{ nullptr };
	ExecutionFrame *m_global_frame{ nullptr };
	std::vector<PyModule *> m_available_modules;
	PyModule *m_module;
	Status m_status{ Status::OK };
	std::string m_exception_message;
	std::string m_entry_script;
	std::shared_ptr<Program> m_program;

  public:
	Interpreter();

	void set_status(Status status) { m_status = status; }
	Status status() const { return m_status; }

	const std::string &exception_message() const { return m_exception_message; }

	template<typename... Ts> void raise_exception(std::string_view p, Ts &&... args)
	{
		m_status = Status::EXCEPTION;
		m_exception_message = fmt::format(p, std::forward<Ts>(args)...);
	}

	template<typename... Ts> void raise_exception(PyObject *exception)
	{
		m_status = Status::EXCEPTION;
		// m_exception_message = "DEPRECATED :(";
		m_current_frame->set_exception(std::move(exception));
	}

	ExecutionFrame *execution_frame() const { return m_current_frame; }
	ExecutionFrame *global_execution_frame() const { return m_global_frame; }

	void set_execution_frame(ExecutionFrame *frame) { m_current_frame = frame; }

	void store_object(const std::string &name, PyObject *obj)
	{
		spdlog::debug("Interpreter::store_object(name={}, obj={}, current_frame={})",
			name,
			obj->to_string(),
			(void *)m_current_frame);
		m_current_frame->put_local(name, obj);
		if (m_current_frame == m_global_frame) { m_current_frame->put_global(name, obj); }
	}

	template<typename PyObjectType, typename... Args>
	PyObject *allocate_object(const std::string &name, Args &&... args)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto obj = heap.allocate<PyObjectType>(std::forward<Args>(args)...)) {
			store_object(name, obj);
			return obj;
		} else {
			return nullptr;
		}
	}

	PyModule *get_imported_module(PyString *) const;

	PyModule *module() const { return m_module; }

	void unwind();

	void setup(std::shared_ptr<Program> program);
	void setup_main_interpreter(std::shared_ptr<Program> program);

	const std::string &entry_script() const { return m_entry_script; }

	const std::shared_ptr<Function> &functions(size_t idx) const;

	PyObject *call(const std::shared_ptr<Function> &, ExecutionFrame *function_frame);

	PyObject *call(PyNativeFunction *native_func, PyTuple *args, PyDict *kwargs);

  private:
	void internal_setup(PyString *name, std::string entry_script, size_t local_registers);
};
