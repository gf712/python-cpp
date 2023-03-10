#include "Warning.hpp"
#include "runtime/PyString.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"

namespace py {

namespace {
	static PyType *s_warning = nullptr;
}

Warning::Warning(PyType *type) : Exception(type) {}

Warning::Warning(PyType *, PyTuple *args) : Exception(s_warning, args) {}

PyResult<Warning *> Warning::create(PyType *type, PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<Warning>(type, args);
	if (!result) { return Err(memory_error(sizeof(Warning))); }
	return Ok(result);
}

PyResult<PyObject *> Warning::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty())
	return Warning::create(const_cast<PyType *>(type), args);
}

PyType *Warning::static_type() const
{
	ASSERT(s_warning)
	return s_warning;
}

PyType *Warning::class_type()
{
	ASSERT(s_warning)
	return s_warning;
}

PyType *Warning::register_type(PyModule *module)
{
	if (!s_warning) {
		s_warning = klass<Warning>(module, "Warning", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("Warning").unwrap(), s_warning);
	}
	return s_warning;
}

}// namespace py
