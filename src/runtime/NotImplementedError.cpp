#include "NotImplementedError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

static PyType *s_not_implemented_error = nullptr;

NotImplementedError::NotImplementedError(PyTuple *args)
	: Exception(s_not_implemented_error->underlying_type(), args)
{}

PyResult<PyObject *> NotImplementedError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_not_implemented_error)
	ASSERT(!kwargs || kwargs->map().empty())
	if (auto result = NotImplementedError::create(args)) {
		return Ok(static_cast<PyObject *>(result));
	} else {
		TODO();
	}
}

PyType *NotImplementedError::type() const
{
	ASSERT(s_not_implemented_error)
	return s_not_implemented_error;
}

PyType *NotImplementedError::register_type(PyModule *module)
{
	if (!s_not_implemented_error) {
		s_not_implemented_error =
			klass<NotImplementedError>(module, "NotImplementedError", Exception::s_exception_type)
				.finalize();
	} else {
		module->add_symbol(
			PyString::create("NotImplementedError").unwrap(), s_not_implemented_error);
	}
	return s_not_implemented_error;
}