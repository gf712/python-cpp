#include "ValueError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

static PyType *s_value_error = nullptr;

ValueError::ValueError(PyTuple *args) : Exception(s_value_error->underlying_type(), args) {}

PyType *ValueError::type() const
{
	ASSERT(s_value_error)
	return s_value_error;
}

PyType *ValueError::register_type(PyModule *module)
{
	if (!s_value_error) {
		s_value_error =
			klass<ValueError>(module, "ValueError", Exception::s_exception_type).finalize();
	}
	return s_value_error;
}