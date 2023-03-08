#include "RuntimeError.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

namespace {
	static PyType *s_runtime_error = nullptr;
}

template<> RuntimeError *as(PyObject *obj)
{
	ASSERT(s_runtime_error)
	if (obj->type() == s_runtime_error) { return static_cast<RuntimeError *>(obj); }
	return nullptr;
}

template<> const RuntimeError *as(const PyObject *obj)
{
	ASSERT(s_runtime_error)
	if (obj->type() == s_runtime_error) { return static_cast<const RuntimeError *>(obj); }
	return nullptr;
}

RuntimeError::RuntimeError(PyType *type) : Exception(type) {}

RuntimeError::RuntimeError(PyTuple *args) : Exception(s_runtime_error->underlying_type(), args) {}

PyResult<RuntimeError *> RuntimeError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate<RuntimeError>(args);
	if (!obj) { return Err(memory_error(sizeof(RuntimeError))); }
	return Ok(obj);
}

PyType *RuntimeError::static_type() const
{
	ASSERT(s_runtime_error)
	return s_runtime_error;
}

PyType *RuntimeError::register_type(PyModule *module)
{
	if (!s_runtime_error) {
		s_runtime_error =
			klass<RuntimeError>(module, "RuntimeError", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("RuntimeError").unwrap(), s_runtime_error);
	}
	spdlog::trace("RuntimeError type @{}", (void *)s_runtime_error);
	return s_runtime_error;
}

}// namespace py
