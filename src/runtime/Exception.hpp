#pragma once

#include "BaseException.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {
class Exception : public BaseException
{
	friend class ::Heap;
	template<typename... Args>
	friend PyObject *exception(const std::string &message, Args &&...args);

  protected:
	Exception(PyType *);

	Exception(PyType *, PyTuple *args);
	Exception(const TypePrototype &type, PyTuple *args);

  private:
	Exception(PyTuple *args);

	static Exception *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<Exception>(args);
	}

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;

	static PyType *class_type();
};


template<typename... Args> inline PyObject *exception(const std::string &message, Args &&...args)
{
	auto *args_tuple =
		PyTuple::create(PyString::create(fmt::format(message, std::forward<Args>(args)...)));
	return Exception::create(args_tuple);
}

}// namespace py
