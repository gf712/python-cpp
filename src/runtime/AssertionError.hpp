#pragma once

#include "BaseException.hpp"
#include "vm/VM.hpp"

class AssertionError : public BaseException
{
	friend class Heap;
	friend PyObject *assertion_error(const std::string &message);

  public:
	std::string to_string() const override { return "AssertionError"; }

  private:
	AssertionError(std::string message) : BaseException("AssertionError", std::move(message)) {}

	static AssertionError *create(const std::string &value)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate_static<AssertionError>(value).get();
	}
};


inline PyObject *assertion_error(const std::string &message)
{
	static PyObject *assertion_error_obj{ nullptr };
	if (!assertion_error_obj) {
		assertion_error_obj = AssertionError::create(message);
	} else {
		static_cast<AssertionError *>(assertion_error_obj)->set_message(message);
	}
	return assertion_error_obj;
}