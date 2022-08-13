#include "StopIteration.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

static PyType *s_stop_iteration = nullptr;

StopIteration::StopIteration(PyTuple *args) : Exception(s_stop_iteration->underlying_type(), args)
{}

PyResult<PyObject *> StopIteration::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_stop_iteration)
	ASSERT(!kwargs || kwargs->map().empty())
	return Ok(StopIteration::create(args));
}

PyType *StopIteration::type() const
{
	ASSERT(s_stop_iteration)
	return s_stop_iteration;
}

PyType *StopIteration::register_type(PyModule *module)
{
	if (!s_stop_iteration) {
		s_stop_iteration =
			klass<StopIteration>(module, "StopIteration", Exception::s_exception_type).finalize();
	} else {
		module->add_symbol(PyString::create("StopIteration").unwrap(), s_stop_iteration);
	}
	return s_stop_iteration;
}