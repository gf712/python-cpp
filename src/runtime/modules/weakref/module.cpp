#include "../Modules.hpp"
#include "PyWeakProxy.hpp"
#include "PyWeakRef.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"

namespace py {
PyModule *weakref_module()
{
	auto *s_weak_ref_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("_weakref").unwrap(),
		PyString::create("").unwrap())
								  .unwrap();

	PyWeakRef::register_type(s_weak_ref_module, "ref");
	PyWeakProxy::register_type(s_weak_ref_module, "proxy");

	return s_weak_ref_module;
}
}// namespace py
