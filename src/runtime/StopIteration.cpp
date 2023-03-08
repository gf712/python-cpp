#include "StopIteration.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

namespace {
	static PyType *s_stop_iteration = nullptr;
}

template<> StopIteration *as(PyObject *obj)
{
	ASSERT(s_stop_iteration);
	if (obj->type() == s_stop_iteration) { return static_cast<StopIteration *>(obj); }
	return nullptr;
}

template<> const StopIteration *as(const PyObject *obj)
{
	ASSERT(s_stop_iteration);
	if (obj->type() == s_stop_iteration) { return static_cast<const StopIteration *>(obj); }
	return nullptr;
}

StopIteration::StopIteration(PyType *type) : Exception(type) {}

StopIteration::StopIteration(PyTuple *args) : Exception(s_stop_iteration->underlying_type(), args)
{}

PyResult<PyObject *> StopIteration::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_stop_iteration)
	ASSERT(!kwargs || kwargs->map().empty())
	return Ok(StopIteration::create(args));
}

PyType *StopIteration::static_type() const
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
}// namespace py
