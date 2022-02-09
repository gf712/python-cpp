#include "RuntimeError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

static PyType *s_runtime_error = nullptr;

RuntimeError::RuntimeError(PyTuple *args) : Exception(s_runtime_error->underlying_type(), args) {}

PyType *RuntimeError::type() const
{
	ASSERT(s_runtime_error)
	return s_runtime_error;
}

PyType *RuntimeError::register_type(PyModule *module)
{
	if (!s_runtime_error) {
		s_runtime_error =
			klass<RuntimeError>(module, "RuntimeError", Exception::s_exception_type).finalize();
	}
	return s_runtime_error;
}