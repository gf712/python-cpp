#include "OSError.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

static py::PyType *s_os_error = nullptr;

namespace py {

template<> OSError *as(PyObject *obj)
{
	ASSERT(s_os_error)
	if (obj->type() == s_os_error) { return static_cast<OSError *>(obj); }
	return nullptr;
}


template<> const OSError *as(const PyObject *obj)
{
	ASSERT(s_os_error)
	if (obj->type() == s_os_error) { return static_cast<const OSError *>(obj); }
	return nullptr;
}

OSError::OSError(PyType *type, PyTuple *args) : Exception(type->underlying_type(), args) {}

OSError::OSError(PyTuple *args) : OSError(s_os_error, args) {}

PyResult<OSError *> OSError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto result = heap.allocate<OSError>(args);
	if (!result) { return Err(memory_error(sizeof(OSError))); }
	return Ok(result);
}

PyResult<PyObject *> OSError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_os_error);
	ASSERT(!kwargs || kwargs->map().empty())
	return OSError::create(args);
}

PyType *OSError::type() const
{
	ASSERT(s_os_error);
	return s_os_error;
}

PyType *OSError::static_type()
{
	ASSERT(s_os_error);
	return s_os_error;
}

PyType *OSError::register_type(PyModule *module)
{
	if (!s_os_error) {
		s_os_error = klass<OSError>(module, "OSError", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("OSError").unwrap(), s_os_error);
	}
	return s_os_error;
}

}// namespace py