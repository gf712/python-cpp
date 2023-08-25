#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {
class AttributeError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *attribute_error(const std::string &message, Args &&...args);

  private:
	AttributeError(PyType *type);

	AttributeError(PyTuple *args);

	static AttributeError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<AttributeError>(args);
	}

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;

	static PyType *class_type();
};


template<typename... Args>
inline BaseException *attribute_error(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	return AttributeError::create(args_tuple.unwrap());
}
}// namespace py
