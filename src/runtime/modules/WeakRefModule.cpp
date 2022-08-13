#include "Modules.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyWeakRef.hpp"

namespace py {
PyModule *weakref_module()
{
	auto *s_weak_ref_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("_weakref").unwrap(),
		PyString::create("").unwrap())
								  .unwrap();

	PyWeakRef::register_type(s_weak_ref_module, "ref");

	return s_weak_ref_module;
}
}// namespace py
