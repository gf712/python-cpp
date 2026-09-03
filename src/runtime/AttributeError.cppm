module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:attribute_error;
import :exception;
import :object;
import std;

export namespace py {
struct Number;
class BaseException;
// Same-module forward declarations: these bind to the definitions in :dict,
// :tuple and :type, so this partition does not have to import them just to
// name them in signatures.
class PyType;
class PyTuple;
class PyDict;

class AttributeError : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_attribute_error(std::string &&);

  private:
	AttributeError(PyType *type);

	AttributeError(PyTuple *args);

	static AttributeError *create(PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;

	static PyType *class_type();
};

BaseException *make_attribute_error(std::string &&message);

template<typename... Args>
inline BaseException *attribute_error(const std::string &message, Args &&...args)
{
	return make_attribute_error(std::vformat(message, std::make_format_args(args...)));
}
}// namespace py
