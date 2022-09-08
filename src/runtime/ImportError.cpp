#include "ImportError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

static PyType *s_import_error = nullptr;

ImportError::ImportError(PyType *type, PyTuple *args) : Exception(type->underlying_type(), args) {}

ImportError::ImportError(PyTuple *args) : Exception(s_import_error->underlying_type(), args) {}

PyResult<PyObject *> ImportError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_import_error)
	ASSERT(!kwargs || kwargs->map().empty())
	if (auto result = ImportError::create(args)) {
		return Ok(static_cast<PyObject *>(result));
	} else {
		TODO();
	}
}

PyType *ImportError::static_type()
{
	ASSERT(s_import_error)
	return s_import_error;
}

PyType *ImportError::type() const
{
	ASSERT(s_import_error)
	return s_import_error;
}

PyType *ImportError::register_type(PyModule *module)
{
	if (!s_import_error) {
		s_import_error =
			klass<ImportError>(module, "ImportError", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("ImportError").unwrap(), s_import_error);
	}
	return s_import_error;
}