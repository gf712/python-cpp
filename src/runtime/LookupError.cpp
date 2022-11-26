#include "LookupError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

namespace {
	static PyType *s_lookup_error = nullptr;
}

LookupError::LookupError(PyType *type, PyTuple *args) : Exception(type->underlying_type(), args) {}

LookupError::LookupError(PyTuple *args) : Exception(s_lookup_error->underlying_type(), args) {}

PyResult<PyObject *> LookupError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_lookup_error)
	ASSERT(!kwargs || kwargs->map().empty())
	if (auto result = LookupError::create(args)) {
		return Ok(static_cast<PyObject *>(result));
	} else {
		TODO();
	}
}

PyType *LookupError::static_type()
{
	ASSERT(s_lookup_error)
	return s_lookup_error;
}

PyType *LookupError::type() const
{
	ASSERT(s_lookup_error)
	return s_lookup_error;
}

PyType *LookupError::register_type(PyModule *module)
{
	if (!s_lookup_error) {
		s_lookup_error =
			klass<LookupError>(module, "LookupError", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("LookupError").unwrap(), s_lookup_error);
	}
	return s_lookup_error;
}
}// namespace py
