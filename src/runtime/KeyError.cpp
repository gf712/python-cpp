#include "KeyError.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

static py::PyType *s_key_error = nullptr;

namespace py {

template<> KeyError *as(PyObject *obj)
{
	ASSERT(s_key_error)
	if (obj->type() == s_key_error) { return static_cast<KeyError *>(obj); }
	return nullptr;
}

template<> const KeyError *as(const PyObject *obj)
{
	ASSERT(s_key_error)
	if (obj->type() == s_key_error) { return static_cast<const KeyError *>(obj); }
	return nullptr;
}

KeyError::KeyError(PyType *type) : Exception(type) {}

KeyError::KeyError(PyTuple *args) : Exception(s_key_error->underlying_type(), args) {}

PyResult<KeyError *> KeyError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto result = heap.allocate<KeyError>(args);
	if (!result) { return Err(memory_error(sizeof(KeyError))); }
	return Ok(result);
}

PyResult<PyObject *> KeyError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_key_error)
	ASSERT(!kwargs || kwargs->map().empty())
	return KeyError::create(args);
}

PyType *KeyError::static_type() const
{
	ASSERT(s_key_error)
	return s_key_error;
}

PyType *KeyError::class_type()
{
	ASSERT(s_key_error)
	return s_key_error;
}

PyType *KeyError::register_type(PyModule *module)
{
	if (!s_key_error) {
		s_key_error = klass<KeyError>(module, "KeyError", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("KeyError").unwrap(), s_key_error);
	}
	return s_key_error;
}

}// namespace py
