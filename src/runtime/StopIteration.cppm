module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:stop_iteration;
import :baseexception;
import :dict;
import :exception;
import :object;
import :tuple;
import :value;
import py.memory;
import std;

export namespace py {
class PyType;

class StopIteration : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	template<typename... Args> friend BaseException *stop_iteration(Args &&...args);

  private:
	StopIteration(PyType *type);

	StopIteration(PyTuple *args);

  public:
	static StopIteration *create(PyTuple *args);

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;

	static PyType *class_type();
};

template<typename... Args> inline BaseException *stop_iteration(Args &&...args)
{
	auto args_tuple = PyTuple::create(std::forward<Args>(args)...);
	if (args_tuple.is_err()) { TODO(); }
	return StopIteration::create(args_tuple.unwrap());
}

}// namespace py
