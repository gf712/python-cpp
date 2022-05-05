#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class ValueError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *value_error(const std::string &message, Args &&... args);

  private:
	ValueError(PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	static PyResult<ValueError *> create(PyTuple *args);

	static PyType *register_type(PyModule *);

	PyType *type() const override;
};

template<typename... Args>
inline BaseException *value_error(const std::string &message, Args &&... args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	return ValueError::create(args_tuple.unwrap()).unwrap();
}

}// namespace py