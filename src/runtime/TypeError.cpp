#include "TypeError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

static PyType *s_type_error = nullptr;

TypeError::TypeError(PyTuple *args) : Exception(s_type_error->underlying_type(), args) {}

PyResult<PyObject *> TypeError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_type_error)
	ASSERT(!kwargs || kwargs->map().empty())
	return Ok(TypeError::create(args));
}

PyType *TypeError::type() const
{
	ASSERT(s_type_error)
	return s_type_error;
}

PyType *TypeError::register_type(PyModule *module)
{
	if (!s_type_error) {
		s_type_error =
			klass<TypeError>(module, "TypeError", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("TypeError").unwrap(), s_type_error);
	}
	return s_type_error;
}