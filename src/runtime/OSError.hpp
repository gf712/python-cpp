#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class OSError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *os_error(const std::string &message, Args &&...args);

  private:
	OSError(PyTuple *args);

  protected:
	OSError(PyType *);
	OSError(PyType *, PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	static PyResult<OSError *> create(PyTuple *args);

	static PyType *register_type(PyModule *);

	PyType *static_type() const override;

	static PyType *class_type();
};

template<typename... Args>
inline BaseException *os_error(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	return OSError::create(args_tuple.unwrap()).unwrap();
}

}// namespace py
