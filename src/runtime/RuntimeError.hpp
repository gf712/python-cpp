#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class RuntimeError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend PyObject *runtime_error(const std::string &message, Args &&... args);

  private:
	RuntimeError(PyTuple *args);

	static RuntimeError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<RuntimeError>(args);
	}

  public:
	static PyType *register_type(PyModule *);

	PyType *type() const override;
};

template<typename... Args> inline PyObject *runtime_error(const std::string &message, Args &&... args)
{
	auto *args_tuple =
		PyTuple::create(PyString::create(fmt::format(message, std::forward<Args>(args)...)));
	return RuntimeError::create(args_tuple);
}

}// namespace py