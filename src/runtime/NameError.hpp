#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

class NameError : public Exception
{
	friend class Heap;
	template<typename... Args>
	friend PyObject *name_error(const std::string &message, Args &&... args);

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


template<typename... Args> inline PyObject *name_error(const std::string &message, Args &&... args)
{
	auto *args_tuple =
		PyTuple::create(PyString::create(fmt::format(message, std::forward<Args>(args)...)));
	return NameError::create(args_tuple);
}