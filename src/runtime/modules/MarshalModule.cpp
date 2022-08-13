#include "Modules.hpp"
#include "runtime/PyDict.hpp"

namespace py {
PyModule *marshal_module()
{
	auto *s_marshal_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("marshal").unwrap(),
		PyString::create("The marshal module!").unwrap())
								 .unwrap();

	return s_marshal_module;
}
}// namespace py
