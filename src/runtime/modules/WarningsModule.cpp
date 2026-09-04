module;
#include "core.hpp"

module py.runtime;


namespace py {
PyModule *warnings_module()
{
	auto *s_warnings_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("_warnings").unwrap(),
		PyString::create("").unwrap())
								  .unwrap();

	s_warnings_module->add_symbol(PyString::create("warn").unwrap(),
		PyNativeFunction::create("warn", [](PyTuple *, PyDict *) {
			// TODO: Implement _warnings.warn
			return Ok(py_none());
		}).unwrap());

	return s_warnings_module;
}
}// namespace py
