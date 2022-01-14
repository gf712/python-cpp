#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class ValueError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend PyObject *value_error(const std::string &message, Args &&... args);

  private:
	ValueError(PyTuple *args);

	static ValueError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<ValueError>(args);
	}

  public:
	static PyType *register_type(PyModule *);

	PyType *type() const override;
};

template<typename... Args> inline PyObject *value_error(const std::string &message, Args &&... args)
{
	auto *args_tuple =
		PyTuple::create(PyString::create(fmt::format(message, std::forward<Args>(args)...)));
	return ValueError::create(args_tuple);
}

}// namespace py