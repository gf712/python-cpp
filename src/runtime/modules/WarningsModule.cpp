#include "Modules.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyType.hpp"
#include "runtime/types/builtin.hpp"

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
