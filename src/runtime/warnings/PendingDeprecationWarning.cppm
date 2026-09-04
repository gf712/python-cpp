module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:pending_deprecation_warning;
import :warning;
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

class PendingDeprecationWarning : public Warning
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_pending_deprecation_warning(std::string &&);

  protected:
	PendingDeprecationWarning(PyType *type);

	PendingDeprecationWarning(PyType *, PyTuple *args);

  private:
	static PyResult<PendingDeprecationWarning *> create(PyType *, PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
	static PyType *class_type();
};

// Defined in warnings/PendingDeprecationWarning.cpp. Keeping the PyString/PyTuple construction and
// the heap allocation out of the interface means this partition no longer has to materialise those
// types just to declare one exception class.
BaseException *make_pending_deprecation_warning(std::string &&message);

template<typename... Args>
inline BaseException *pending_deprecation_warning(const std::string &message, Args &&...args)
{
	return make_pending_deprecation_warning(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
