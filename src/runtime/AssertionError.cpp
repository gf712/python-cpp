#include "AssertionError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

static PyType *s_assertion_error = nullptr;

AssertionError::AssertionError(PyTuple *args)
	: Exception(s_assertion_error->underlying_type(), args)
{}

PyObject *AssertionError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_assertion_error)
	ASSERT(!kwargs || kwargs->map().empty())
	return AssertionError::create(args);
}

PyType *AssertionError::type() const
{
	ASSERT(s_assertion_error)
	return s_assertion_error;
}

PyType *AssertionError::this_type()
{
	ASSERT(s_assertion_error)
	return s_assertion_error;
}

std::string AssertionError::to_string() const
{
	return what();
}

PyType *AssertionError::register_type(PyModule *module)
{
	if (!s_assertion_error) {
		s_assertion_error =
			klass<AssertionError>(module, "AssertionError", Exception::s_exception_type).finalize();
	}
	return s_assertion_error;
}