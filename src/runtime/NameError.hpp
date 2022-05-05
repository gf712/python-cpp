#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class NameError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *name_error(const std::string &message, Args &&... args);

  private:
	NameError(PyTuple *args);

	static NameError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<NameError>(args);
	}

  public:
	static PyType *register_type(PyModule *);

	PyType *type() const override;
};


template<typename... Args>
inline BaseException *name_error(const std::string &message, Args &&... args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	return NameError::create(args_tuple.unwrap());
}

}// namespace py