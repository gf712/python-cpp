#include "Exception.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

PyType *Exception::s_exception_type = nullptr;

Exception::Exception(PyTuple *args) : BaseException(s_exception_type->underlying_type(), args) {}

Exception::Exception(const TypePrototype &type, PyTuple *args) : BaseException(type, args) {}

PyType *Exception::type() const
{
	ASSERT(s_exception_type)
	return s_exception_type;
}

PyType *Exception::register_type(PyModule *module)
{
	if (!s_exception_type) {
		s_exception_type =
			klass<Exception>(module, "Exception", BaseException::s_base_exception_type).finalize();
	}
	return s_exception_type;
}