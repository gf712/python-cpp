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
	friend BaseException *attribute_error(const std::string &message, Args &&... args);

  private:
	AttributeError(PyTuple *args);

	static AttributeError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<AttributeError>(args);
	}

  public:
	static PyType *register_type(PyModule *);

	PyType *type() const override;

	static PyType *static_type();
};


template<typename... Args>
inline BaseException *attribute_error(const std::string &message, Args &&... args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.template unwrap_as<PyString>());
	ASSERT(args_tuple.is_ok())
	return AttributeError::create(args_tuple.template unwrap_as<PyTuple>());
}
}// namespace py