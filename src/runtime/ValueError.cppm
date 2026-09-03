module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:value_error;
import :baseexception;
import :dict;
import :exception;
import :object;
import :string;
import :tuple;
import :value;
import py.memory;
import std;

export namespace py {
class PyType;

class ValueError : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_value_error(std::string &&);

  private:
	ValueError(PyType *type);

	ValueError(PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	static PyResult<ValueError *> create(PyTuple *args);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;

	static PyType *class_type();
};

// Defined in ValueError.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_value_error(std::string &&message);

template<typename... Args>
inline BaseException *value_error(const std::string &message, Args &&...args)
{
	return make_value_error(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
