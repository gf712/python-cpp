#pragma once

#include "runtime/Exception.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class Warning : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *warning(const std::string &message, Args &&...args);

  protected:
	Warning(PyType *type);

	Warning(PyType *, PyTuple *args);

  private:
	static PyResult<Warning *> create(PyType *, PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static PyType *register_type(PyModule *);

	PyType *static_type() const override;
	static PyType *class_type();
};

template<typename... Args> inline BaseException *warning(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	return Warning::create(Warning::class_type(), args_tuple.unwrap()).unwrap();
}

}// namespace py
