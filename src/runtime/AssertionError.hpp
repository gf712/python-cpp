#pragma once

#include "Exception.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class AssertionError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *assertion_error(const std::string &message, Args &&...args);

  private:
	AssertionError(PyType *t);

	AssertionError(PyTuple *args);

	static PyResult<AssertionError *> create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		auto result = heap.allocate<AssertionError>(args);
		if (!result) { return Err(memory_error(sizeof(AssertionError))); }
		return Ok(result);
	}

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyType *static_type() const override;

	static PyType *this_type();

	std::string to_string() const override;
};

template<typename... Args>
inline BaseException *assertion_error(const std::string &message, Args &&...args)
{
	auto *args_tuple =
		PyTuple::create(PyString::create(fmt::format(message, std::forward<Args>(args)...)));
	return AssertionError::create(args_tuple).unwrap();
}
}// namespace py
