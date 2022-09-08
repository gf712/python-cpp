#include "ModuleNotFoundError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

namespace {
	static PyType *s_module_not_found_error = nullptr;
}

ModuleNotFoundError::ModuleNotFoundError(PyTuple *args, PyObject *name, PyObject *path)
	: ImportError(s_module_not_found_error, args), m_name(name), m_path(path)
{}

PyResult<PyObject *> ModuleNotFoundError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_module_not_found_error);
	auto name = [kwargs]() -> PyResult<PyObject *> {
		auto *name = PyString::create("name").unwrap();
		if (kwargs->map().contains(name)) { return PyObject::from(kwargs->map().at(name)); }
		return Ok(py_none());
	}();
	if (name.is_err()) return name;

	auto path = [kwargs]() -> PyResult<PyObject *> {
		auto *path = PyString::create("path").unwrap();
		if (kwargs->map().contains(path)) { return PyObject::from(kwargs->map().at(path)); }
		return Ok(py_none());
	}();
	if (path.is_err()) return path;

	if (auto result = ModuleNotFoundError::create(args, name.unwrap(), path.unwrap())) {
		return Ok(static_cast<PyObject *>(result));
	} else {
		TODO();
		return Err(nullptr);
	}
}

PyResult<int32_t> ModuleNotFoundError::__init__(PyTuple *, PyDict *) { return Ok(0); }

PyType *ModuleNotFoundError::type() const
{
	ASSERT(s_module_not_found_error)
	return s_module_not_found_error;
}

PyType *ModuleNotFoundError::register_type(PyModule *module)
{
	if (!s_module_not_found_error) {
		s_module_not_found_error =
			klass<ModuleNotFoundError>(module, "ModuleNotFoundError", ImportError::static_type())
				.finalize();
	} else {
		module->add_symbol(
			PyString::create("ModuleNotFoundError").unwrap(), s_module_not_found_error);
	}
	return s_module_not_found_error;
}
}// namespace py