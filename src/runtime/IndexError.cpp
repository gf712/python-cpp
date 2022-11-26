#include "IndexError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

namespace {
	static PyType *s_index_error = nullptr;
}

IndexError::IndexError(PyTuple *args) : LookupError(s_index_error, args) {}

PyResult<PyObject *> IndexError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_index_error)
	ASSERT(!kwargs || kwargs->map().empty())
	if (auto result = IndexError::create(args)) {
		return Ok(static_cast<PyObject *>(result));
	} else {
		TODO();
	}
}

PyType *IndexError::static_type()
{
	ASSERT(s_index_error)
	return s_index_error;
}

PyType *IndexError::type() const
{
	ASSERT(s_index_error)
	return s_index_error;
}

PyType *IndexError::register_type(PyModule *module)
{
	if (!s_index_error) {
		s_index_error =
			klass<IndexError>(module, "IndexError", LookupError::static_type()).finalize();
	} else {
		module->add_symbol(PyString::create("IndexError").unwrap(), s_index_error);
	}
	return s_index_error;
}

}// namespace py
