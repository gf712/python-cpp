#pragma once

#include "BaseException.hpp"
#include "vm/VM.hpp"


class ValueError : public BaseException
{
	friend class Heap;
	template<typename... Args>
	friend PyObject *value_error(const std::string &message, Args &&... args);

  public:
	std::string to_string() const override { return "ValueError"; }

  private:
	ValueError(std::string message) : BaseException("ValueError", std::move(message)) {}

	static ValueError *create(const std::string &value)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate_static<ValueError>(value).get();
	}
};


template<typename... Args> inline PyObject *value_error(const std::string &message, Args &&... args)
{
	static PyObject *value_error_{ nullptr };
	if (!value_error_) {
		value_error_ = ValueError::create(fmt::format(message, std::forward<Args>(args)...));
	} else {
		static_cast<ValueError *>(value_error_)
			->set_message(fmt::format(message, std::forward<Args>(args)...));
	}
	return value_error_;
}