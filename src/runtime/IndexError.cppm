module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:index_error;
import :lookup_error;
import :baseexception;
import :dict;
import :object;
import :string;
import :tuple;
import :value;
import py.memory;
import std;

export namespace py {
class PyType;

class IndexError : public LookupError
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_index_error(std::string &&);

  private:
	IndexError(PyType *type);

	IndexError(PyTuple *msg);

	static IndexError *create(PyTuple *args);

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyType *static_type() const override;
	static PyType *class_type();
};

// Defined in IndexError.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_index_error(std::string &&message);

template<typename... Args>
inline BaseException *index_error(const std::string &message, Args &&...args)
{
	return make_index_error(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
