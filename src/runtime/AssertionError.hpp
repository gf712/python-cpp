#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

class AssertionError : public Exception
{
	friend class Heap;
	template<typename... Args>
	friend PyObject *assertion_error(const std::string &message, Args &&... args);

  private:
	AssertionError(PyTuple *args);

	static AssertionError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<AssertionError>(args);
	}

  public:
	static PyType *register_type(PyModule *);

	static PyObject *__new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyType *type() const override;

	static PyType *this_type();

	std::string to_string() const override;
};

template<typename... Args>
inline PyObject *assertion_error(const std::string &message, Args &&... args)
{
	auto *args_tuple =
		PyTuple::create(PyString::create(fmt::format(message, std::forward<Args>(args)...)));
	return AssertionError::create(args_tuple);
}