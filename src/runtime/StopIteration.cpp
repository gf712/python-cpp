#include "StopIteration.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

static PyType *s_stop_iteration = nullptr;

StopIteration::StopIteration(PyTuple *args) : Exception(s_stop_iteration->underlying_type(), args)
{}

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
	}
	return s_stop_iteration;
}