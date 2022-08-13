#include "Modules.hpp"
#include "runtime/PyDict.hpp"

namespace py {
PyModule *warnings_module()
{
	auto *s_warnings_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("_warnings").unwrap(),
		PyString::create("").unwrap())
								  .unwrap();

	return s_warnings_module;
}
}// namespace py
