#include "ValueError.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

static py::PyType *s_value_error = nullptr;

namespace py {

template<> ValueError *as(PyObject *obj)
{
	ASSERT(s_value_error)
	if (obj->type() == s_value_error) { return static_cast<ValueError *>(obj); }
	return nullptr;
}


template<> const ValueError *as(const PyObject *obj)
{
	ASSERT(s_value_error)
	if (obj->type() == s_value_error) { return static_cast<const ValueError *>(obj); }
	return nullptr;
}

ValueError::ValueError(PyTuple *args) : Exception(s_value_error->underlying_type(), args) {}

PyResult<ValueError *> ValueError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto result = heap.allocate<ValueError>(args);
	if (!result) { return Err(memory_error(sizeof(ValueError))); }
	return Ok(result);
}

PyResult<PyObject *> ValueError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_value_error)
	ASSERT(!kwargs || kwargs->map().empty())
	return ValueError::create(args);
}

PyType *ValueError::type() const
{
	ASSERT(s_value_error)
	return s_value_error;
}

PyType *ValueError::register_type(PyModule *module)
{
	if (!s_value_error) {
		s_value_error =
			klass<ValueError>(module, "ValueError", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("ValueError").unwrap(), s_value_error);
	}
	return s_value_error;
}

}// namespace py