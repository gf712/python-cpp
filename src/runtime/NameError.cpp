#include "NameError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

static PyType *s_name_error = nullptr;

NameError::NameError(PyType *type) : Exception(type) {}

NameError::NameError(PyTuple *args) : Exception(s_name_error->underlying_type(), args) {}

PyType *NameError::static_type() const
{
	ASSERT(s_name_error)
	return s_name_error;
}

PyType *NameError::register_type(PyModule *module)
{
	if (!s_name_error) {
		s_name_error =
			klass<Exception>(module, "NameError", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("NameError").unwrap(), s_name_error);
	}
	return s_name_error;
}
