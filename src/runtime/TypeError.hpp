#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class TypeError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *type_error(const std::string &message, Args &&... args);

  private:
	TypeError(PyTuple *args);

	static TypeError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<TypeError>(args);
	}

  public:
	static PyType *register_type(PyModule *);

	PyType *type() const override;
};

template<typename... Args>
inline BaseException *type_error(const std::string &message, Args &&... args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.template unwrap_as<PyString>());
	ASSERT(args_tuple.is_ok())
	return TypeError::create(args_tuple.template unwrap_as<PyTuple>());
}

}// namespace py