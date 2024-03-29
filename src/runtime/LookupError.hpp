#pragma once

#include "Exception.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class LookupError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *lookup_error(const std::string &message, Args &&...args);

  protected:
	LookupError(PyType *);

	LookupError(PyType *, PyTuple *msg);

	LookupError(TypePrototype &, PyTuple *msg);

  private:
	LookupError(PyTuple *msg);

	static LookupError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<LookupError>(args);
	}

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyType *static_type() const override;
	static PyType *class_type();
};

template<typename... Args>
inline BaseException *lookup_error(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	return LookupError::create(args_tuple.unwrap());
}

}// namespace py
