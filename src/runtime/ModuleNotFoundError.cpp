#include "ModuleNotFoundError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

ModuleNotFoundError::ModuleNotFoundError(PyType *type) : ImportError(type, nullptr) {}

ModuleNotFoundError::ModuleNotFoundError(PyTuple *args, PyObject *name, PyObject *path)
	: ImportError(types::BuiltinTypes::the().module_not_found_error(), args), m_name(name),
	  m_path(path)
{}

PyResult<PyObject *> ModuleNotFoundError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::module_not_found_error());
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

PyType *ModuleNotFoundError::static_type() const
{
	ASSERT(types::module_not_found_error());
	return types::module_not_found_error();
}

namespace {

	std::once_flag module_not_found_error_flag;

	std::unique_ptr<TypePrototype> register_module_not_found_error()
	{
		return std::move(
			klass<ModuleNotFoundError>("ModuleNotFoundError", ImportError::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> ModuleNotFoundError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(
			module_not_found_error_flag, []() { type = register_module_not_found_error(); });
		return std::move(type);
	};
}

}// namespace py
