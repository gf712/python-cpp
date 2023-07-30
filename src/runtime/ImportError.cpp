#include "ImportError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

static PyType *s_import_error = nullptr;

ImportError::ImportError(PyType *type) : Exception(type->underlying_type(), nullptr) {}

ImportError::ImportError(PyType *type, PyTuple *args) : Exception(type->underlying_type(), args) {}

ImportError::ImportError(PyTuple *args) : Exception(s_import_error->underlying_type(), args) {}

PyResult<PyObject *> ImportError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_import_error)
	if (auto result = ImportError::create(args)) {
		if (kwargs) {
			if (auto it = kwargs->map().find(String{ "name" }); it != kwargs->map().end()) {
				result->m_name = PyObject::from(it->second).unwrap();
			}
		}

		return Ok(static_cast<PyObject *>(result));
	} else {
		TODO();
	}
}

PyResult<int32_t> ImportError::__init__(PyTuple *args, PyDict *kwargs)
{
	m_args = args;
	if (kwargs) {
		if (auto it = kwargs->map().find(String{ "name" }); it != kwargs->map().end()) {
			m_name = PyObject::from(it->second).unwrap();
		}
	}
	return Ok(1);
}

PyType *ImportError::class_type()
{
	ASSERT(s_import_error)
	return s_import_error;
}

PyType *ImportError::static_type() const
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

void ImportError::visit_graph(Visitor &visitor)
{
	Exception::visit_graph(visitor);
	if (m_name) visitor.visit(*m_name);
}
