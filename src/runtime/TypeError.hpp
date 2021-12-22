#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"


class TypeError : public Exception
{
	friend class Heap;
	template<typename... Args>
	friend PyObject *type_error(const std::string &message, Args &&... args);

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

template<typename... Args> inline PyObject *type_error(const std::string &message, Args &&... args)
{
	auto *args_tuple =
		PyTuple::create(PyString::create(fmt::format(message, std::forward<Args>(args)...)));
	return TypeError::create(args_tuple);
}