#pragma once

#include "BaseException.hpp"
#include "bytecode/VM.hpp"


class TypeError : public BaseException
{
	friend class Heap;
	template<typename... Args>
	friend std::shared_ptr<PyObject> type_error(const std::string &message, Args &&... args);

  public:
	std::string to_string() const override { return "TypeError"; }

  private:
	TypeError(std::string message) : BaseException("TypeError", std::move(message)) {}

	static std::shared_ptr<TypeError> create(const std::string &value)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate_static<TypeError>(value);
	}
};


template<typename... Args>
inline std::shared_ptr<PyObject> type_error(const std::string &message, Args &&... args)
{
	static std::shared_ptr<PyObject> type_error_{ nullptr };
	if (!type_error_) {
		type_error_ = TypeError::create(fmt::format(message, std::forward<Args>(args)...));
	} else {
		std::static_pointer_cast<TypeError>(type_error_)
			->set_message(fmt::format(message, std::forward<Args>(args)...));
	}
	return type_error_;
}