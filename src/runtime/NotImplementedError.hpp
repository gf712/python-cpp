#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class NotImplementedError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *not_implemented_error(const std::string &message, Args &&...args);

  private:
	NotImplementedError(PyType *);

	NotImplementedError(PyTuple *args);

	static NotImplementedError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<NotImplementedError>(args);
	}

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static PyType *register_type(PyModule *);

	PyType *static_type() const override;
};


template<typename... Args>
inline BaseException *not_implemented_error(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	return NotImplementedError::create(args_tuple.unwrap());
}

}// namespace py
