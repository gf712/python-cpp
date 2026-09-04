module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:warning;
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

class Warning : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	template<typename... Args>
	friend BaseException *warning(const std::string &message, Args &&...args);

  protected:
	Warning(PyType *type);

	Warning(PyType *, PyTuple *args);

  private:
	static PyResult<Warning *> create(PyType *, PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
	static PyType *class_type();
};

template<typename... Args> inline BaseException *warning(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(std::vformat(message, std::make_format_args(args...)));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return Warning::create(Warning::class_type(), args_tuple.unwrap()).unwrap();
}

}// namespace py
