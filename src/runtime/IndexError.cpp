#include "IndexError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

IndexError::IndexError(PyType *type) : LookupError(type, nullptr) {}

IndexError::IndexError(PyTuple *args) : LookupError(types::BuiltinTypes::the().index_error(), args)
{}

PyResult<PyObject *> IndexError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::index_error());
	ASSERT(!kwargs || kwargs->map().empty())
	if (auto result = IndexError::create(args)) {
		return Ok(static_cast<PyObject *>(result));
	} else {
		TODO();
	}
}

PyType *IndexError::class_type()
{
	ASSERT(types::index_error());
	return types::index_error();
}

PyType *IndexError::static_type() const
{
	ASSERT(types::index_error());
	return types::index_error();
}

namespace {

	std::once_flag index_error_flag;

	std::unique_ptr<TypePrototype> register_index_error()
	{
		return std::move(klass<IndexError>("IndexError", LookupError::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> IndexError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(index_error_flag, []() { type = register_index_error(); });
		return std::move(type);
	};
}

}// namespace py
