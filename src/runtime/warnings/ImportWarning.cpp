#include "ImportWarning.hpp"
#include "runtime/PyString.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"

namespace py {

namespace {
	static PyType *s_import_warning = nullptr;
}

ImportWarning::ImportWarning(PyType *type) : Warning(type) {}

ImportWarning::ImportWarning(PyType *type, PyTuple *args) : Warning(type, args) {}

PyResult<ImportWarning *> ImportWarning::create(PyType *type, PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<ImportWarning>(type, args);
	if (!result) { return Err(memory_error(sizeof(ImportWarning))); }
	return Ok(result);
}

PyResult<PyObject *> ImportWarning::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty())
	return ImportWarning::create(const_cast<PyType *>(type), args);
}

PyType *ImportWarning::static_type() const
{
	ASSERT(s_import_warning)
	return s_import_warning;
}

PyType *ImportWarning::class_type()
{
	ASSERT(s_import_warning)
	return s_import_warning;
}

PyType *ImportWarning::register_type(PyModule *module)
{
	if (!s_import_warning) {
		s_import_warning =
			klass<ImportWarning>(module, "ImportWarning", Warning::class_type()).finalize();
	} else {
		module->add_symbol(PyString::create("ImportWarning").unwrap(), s_import_warning);
	}
	return s_import_warning;
}

}// namespace py
