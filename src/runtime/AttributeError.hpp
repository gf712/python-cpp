#pragma once

#include "BaseException.hpp"
#include "vm/VM.hpp"

class AttributeError : public BaseException
{
	friend class Heap;
	template<typename... Args>
	friend PyObject *attribute_error(const std::string &message, Args &&... args);

  public:
	std::string to_string() const override { return "AttributeError"; }

  private:
	AttributeError(std::string message) : BaseException("AttributeError", std::move(message)) {}

	static AttributeError *create(const std::string &value)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate_static<AttributeError>(value).get();
	}
};


template<typename... Args>
inline PyObject *attribute_error(const std::string &message, Args &&... args)
{
	static PyObject *attribute_error_obj{ nullptr };
	if (!attribute_error_obj) {
		attribute_error_obj =
			AttributeError::create(fmt::format(message, std::forward<Args>(args)...));
	} else {
		static_cast<AttributeError *>(attribute_error_obj)
			->set_message(fmt::format(message, std::forward<Args>(args)...));
	}
	return attribute_error_obj;
}