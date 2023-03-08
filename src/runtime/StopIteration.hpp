#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class StopIteration : public Exception
{
	friend class ::Heap;
	template<typename... Args> friend BaseException *stop_iteration(Args &&...args);

  private:
	StopIteration(PyType* type);

	StopIteration(PyTuple *args);

  public:
	static StopIteration *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<StopIteration>(args);
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static PyType *register_type(PyModule *);

	PyType *static_type() const override;
};

template<typename... Args> inline BaseException *stop_iteration(Args &&...args)
{
	auto args_tuple = PyTuple::create(std::forward<Args>(args)...);
	if (args_tuple.is_err()) { TODO(); }
	return StopIteration::create(args_tuple.unwrap());
}

}// namespace py
