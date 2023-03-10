#include "Exception.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

PyType *Exception::s_exception_type = nullptr;

Exception::Exception(PyType *t) : BaseException(t->underlying_type(), nullptr) {}

Exception::Exception(PyType *t, PyTuple *args) : BaseException(t, args) {}

Exception::Exception(PyTuple *args) : BaseException(s_exception_type->underlying_type(), args) {}

Exception::Exception(const TypePrototype &type, PyTuple *args) : BaseException(type, args) {}

PyType *Exception::static_type() const
{
	ASSERT(s_exception_type)
	return s_exception_type;
}

PyType *Exception::class_type()
{
	ASSERT(s_exception_type)
	return s_exception_type;
}

PyType *Exception::register_type(PyModule *module)
{
	if (!s_exception_type) {
		s_exception_type =
			klass<Exception>(module, "Exception", BaseException::class_type()).finalize();
	} else {
		module->add_symbol(PyString::create("Exception").unwrap(), s_exception_type);
	}
	return s_exception_type;
}
