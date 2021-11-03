#pragma once

#include "BaseException.hpp"
#include "vm/VM.hpp"


class TypeError : public BaseException
{
	friend class Heap;
	template<typename... Args>
	friend PyObject *type_error(const std::string &message, Args &&... args);

  public:
	std::string to_string() const override { return "TypeError"; }

  private:
	TypeError(std::string message) : BaseException("TypeError", std::move(message)) {}

	static TypeError *create(const std::string &value)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate_static<TypeError>(value).get();
	}
};


template<typename... Args> inline PyObject *type_error(const std::string &message, Args &&... args)
{
	static PyObject *type_error_{ nullptr };
	if (!type_error_) {
		type_error_ = TypeError::create(fmt::format(message, std::forward<Args>(args)...));
	} else {
		static_cast<TypeError *>(type_error_)
			->set_message(fmt::format(message, std::forward<Args>(args)...));
	}
	return type_error_;
}