#include "ImportError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

ImportError::ImportError(PyType *type) : ImportError(type->underlying_type(), nullptr) {}

ImportError::ImportError(PyType *type, PyTuple *args) : ImportError(type->underlying_type(), args)
{}

ImportError::ImportError(TypePrototype &type, PyTuple *args) : Exception(type, args) {}

ImportError::ImportError(PyTuple *args)
	: ImportError(types::BuiltinTypes::the().import_error(), args)
{}

PyResult<PyObject *> ImportError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::import_error());
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
	ASSERT(types::import_error());
	return types::import_error();
}

PyType *ImportError::static_type() const
{
	ASSERT(types::import_error());
	return types::import_error();
}

void ImportError::visit_graph(Visitor &visitor)
{
	Exception::visit_graph(visitor);
	if (m_name) visitor.visit(*m_name);
}

namespace {

	std::once_flag import_error_flag;

	std::unique_ptr<TypePrototype> register_import_error()
	{
		return std::move(klass<ImportError>("ImportError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> ImportError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(import_error_flag, []() { type = register_import_error(); });
		return std::move(type);
	};
}

}// namespace py
