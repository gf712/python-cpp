#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

namespace {
	static PyType *s_memory_error = nullptr;
}

template<> MemoryError *as(PyObject *obj)
{
	ASSERT(s_memory_error)
	if (obj->type() == s_memory_error) { return static_cast<MemoryError *>(obj); }
	return nullptr;
}

template<> const MemoryError *as(const PyObject *obj)
{
	ASSERT(s_memory_error)
	if (obj->type() == s_memory_error) { return static_cast<const MemoryError *>(obj); }
	return nullptr;
}

MemoryError::MemoryError(PyType *type) : Exception(type->underlying_type(), nullptr) {}

MemoryError::MemoryError(PyTuple *args) : Exception(s_memory_error->underlying_type(), args) {}

PyResult<PyObject *> MemoryError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_memory_error)
	ASSERT(!kwargs || kwargs->map().empty())
	if (auto result = MemoryError::create(args); result.is_ok()) {
		return Ok(static_cast<PyObject *>(result.unwrap()));
	} else {
		return Err(result.unwrap_err());
	}
}

PyType *MemoryError::static_type() const
{
	ASSERT(s_memory_error)
	return s_memory_error;
}

PyType *MemoryError::this_type()
{
	ASSERT(s_memory_error)
	return s_memory_error;
}

std::string MemoryError::to_string() const { return what(); }

PyType *MemoryError::register_type(PyModule *module)
{
	if (!s_memory_error) {
		s_memory_error =
			klass<MemoryError>(module, "MemoryError", Exception::s_exception_type).finalize();
	}
	return s_memory_error;
}

}// namespace py
