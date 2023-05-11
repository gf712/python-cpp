#include "SyntaxError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

static PyType *s_syntax_error = nullptr;

SyntaxError::SyntaxError(PyType *type) : Exception(type) {}

SyntaxError::SyntaxError(PyTuple *args) : Exception(s_syntax_error->underlying_type(), args) {}

PyResult<PyObject *> SyntaxError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_syntax_error)
	ASSERT(!kwargs || kwargs->map().empty())
	return Ok(SyntaxError::create(args));
}

PyType *SyntaxError::static_type() const
{
	ASSERT(s_syntax_error)
	return s_syntax_error;
}

PyType *SyntaxError::register_type(PyModule *module)
{
	if (!s_syntax_error) {
		s_syntax_error =
			klass<SyntaxError>(module, "SyntaxError", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("SyntaxError").unwrap(), s_syntax_error);
	}
	return s_syntax_error;
}
