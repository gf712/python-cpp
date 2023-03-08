#include "AttributeError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

static PyType *s_attribute_error = nullptr;

AttributeError::AttributeError(PyType *type) : Exception(type->underlying_type(), nullptr) {}

AttributeError::AttributeError(PyTuple *args)
	: Exception(s_attribute_error->underlying_type(), args)
{}

PyType *AttributeError::static_type() const
{
	ASSERT(s_attribute_error)
	return s_attribute_error;
}

PyType *AttributeError::register_type(PyModule *module)
{
	if (!s_attribute_error) {
		s_attribute_error =
			klass<AttributeError>(module, "AttributeError", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("AttributeError").unwrap(), s_attribute_error);
	}
	return s_attribute_error;
}

PyType *AttributeError::class_type()
{
	ASSERT(s_attribute_error)
	return s_attribute_error;
}
