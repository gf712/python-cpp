module;
#include "memory/allocate.hpp"

export module py.runtime:exception;
import :baseexception;
import py.memory;

export namespace py {
class PyString;
class PyType;
class Exception : public BaseException
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_exception(std::string &&);

  protected:
	Exception(PyType *);

	Exception(PyType *, PyTuple *args);
	Exception(const TypePrototype &type, PyTuple *args);

  private:
	Exception(PyTuple *args);

	static Exception *create(PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;

	static PyType *class_type();
};

// Defined in Exception.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_exception(std::string &&message);

template<typename... Args>
inline BaseException *exception(const std::string &message, Args &&...args)
{
	return make_exception(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
