#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class SyntaxError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *syntax_error(const std::string &message, Args &&...args);

  private:
	SyntaxError(PyType *type);

	SyntaxError(PyTuple *args);

	static SyntaxError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<SyntaxError>(args);
	}

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
};

template<typename... Args>
inline BaseException *syntax_error(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	return SyntaxError::create(args_tuple.unwrap());
}

}// namespace py
