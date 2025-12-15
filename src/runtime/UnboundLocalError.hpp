#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class UnboundLocalError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *unbound_local_error(const std::string &message, Args &&...args);

  private:
	UnboundLocalError(PyType *type);

	UnboundLocalError(PyTuple *args);

	static UnboundLocalError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<UnboundLocalError>(args);
	}

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
};

template<typename... Args>
inline BaseException *unbound_local_error(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return UnboundLocalError::create(args_tuple.unwrap());
}

}// namespace py
