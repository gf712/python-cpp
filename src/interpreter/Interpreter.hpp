#pragma once

#include "ExecutionFrame.hpp"
#include "bytecode/VM.hpp"

#include <string>
#include <string_view>

class Interpreter
{
  public:
	enum class Status { OK, EXCEPTION };

  private:
	std::shared_ptr<ExecutionFrame> m_current_frame{ nullptr };
	Status m_status{ Status::OK };
	std::string m_exception_message;

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

	template<typename... Ts> void raise_exception(std::shared_ptr<PyObject> exception)
	{
		m_status = Status::EXCEPTION;
		m_exception_message = "DEPRECATED :(";
		m_current_frame->set_exception(std::move(exception));
	}

	const std::shared_ptr<ExecutionFrame> &execution_frame() const { return m_current_frame; }

	void set_execution_frame(std::shared_ptr<ExecutionFrame> frame)
	{
		m_current_frame = std::move(frame);
	}

	std::shared_ptr<PyObject> fetch_object(const std::string &name) const
	{
		return m_current_frame->fetch_object(name);
	}

	void store_object(const std::string &name, const std::shared_ptr<PyObject> &obj)
	{
		m_current_frame->put_object(name, obj);
	}

	template<typename PyObjectType, typename... Args>
	std::shared_ptr<PyObject> allocate_object(const std::string &name, Args &&... args)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto obj = heap.allocate<PyObjectType>(std::forward<Args>(args)...)) {
			m_current_frame->put_object(name, obj);
			return obj;
		} else {
			return nullptr;
		}
	}

	void unwind();

	void setup();
};
