#pragma once

#include "BaseException.hpp"
#include "bytecode/VM.hpp"

class StopIterationException : public BaseException
{
	friend class Heap;
	friend std::shared_ptr<PyObject> stop_iteration(const std::string &message);

  public:
	std::string to_string() const override { return "StopIterationException"; }

  private:
	StopIterationException(std::string message) : BaseException("StopIteration", std::move(message))
	{}

	static std::shared_ptr<StopIterationException> create(const std::string &value)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate_static<StopIterationException>(value);
	}
};


inline std::shared_ptr<PyObject> stop_iteration(const std::string &message)
{
	static std::shared_ptr<PyObject> stop_iter_obj{ nullptr };
	if (!stop_iter_obj) {
		stop_iter_obj = StopIterationException::create(message);
	} else {
		std::static_pointer_cast<StopIterationException>(stop_iter_obj)->set_message(message);
	}
	return stop_iter_obj;
}