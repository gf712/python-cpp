#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

class AttributeError : public Exception
{
	friend class Heap;
	template<typename... Args>
	friend PyObject *attribute_error(const std::string &message, Args &&... args);

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
};


template<typename... Args>
inline PyObject *attribute_error(const std::string &message, Args &&... args)
{
	auto *args_tuple =
		PyTuple::create(PyString::create(fmt::format(message, std::forward<Args>(args)...)));
	return AttributeError::create(args_tuple);
}