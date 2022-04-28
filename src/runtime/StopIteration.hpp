#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class StopIteration : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *stop_iteration(const std::string &message, Args &&... args);

  private:
	StopIteration(PyTuple *args);

	static StopIteration *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<StopIteration>(args);
	}

  public:
	static PyType *register_type(PyModule *);

	PyType *type() const override;
};

template<typename... Args>
inline BaseException *stop_iteration(const std::string &message, Args &&... args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.template unwrap_as<PyString>());
	if (args_tuple.is_err()) { TODO(); }
	return StopIteration::create(args_tuple.template unwrap_as<PyTuple>());
}

}// namespace py