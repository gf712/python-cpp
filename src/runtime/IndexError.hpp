#pragma once

#include "Exception.hpp"
#include "LookupError.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class IndexError : public LookupError
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *index_error(const std::string &message, Args &&...args);

  private:
	IndexError(PyType *type);

	IndexError(PyTuple *msg);

	static IndexError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<IndexError>(args);
	}

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyType *static_type() const override;
	static PyType *class_type();
};

template<typename... Args>
inline BaseException *index_error(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return IndexError::create(args_tuple.unwrap());
}

}// namespace py
