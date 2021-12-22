#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

class StopIteration : public Exception
{
	friend class Heap;
	template<typename... Args>
	friend PyObject *stop_iteration(const std::string &message, Args &&... args);

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
inline PyObject *stop_iteration(const std::string &message, Args &&... args)
{
	auto *args_tuple =
		PyTuple::create(PyString::create(fmt::format(message, std::forward<Args>(args)...)));
	return StopIteration::create(args_tuple);
}