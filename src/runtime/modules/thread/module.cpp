#include "../Modules.hpp"
#include "Lock.hpp"
#include "RLock.hpp"
#include "runtime/PyDict.hpp"

namespace py {
PyModule *thread_module()
{
	auto *s_thread_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("_thread").unwrap(),
		PyString::create("").unwrap())
								.unwrap();

	(void)Lock::register_type(s_thread_module);
	(void)RLock::register_type(s_thread_module);

	auto *allocate_lock =
		PyNativeFunction::create("allocate_lock", [](PyTuple *, PyDict *) -> PyResult<PyObject *> {
			return Lock::create();
		}).unwrap();

	s_thread_module->add_symbol(PyString::create("allocate_lock").unwrap(), allocate_lock);
	s_thread_module->add_symbol(PyString::create("allocate").unwrap(), allocate_lock);

	s_thread_module->add_symbol(PyString::create("get_ident").unwrap(),
		PyNativeFunction::create("get_ident", [](PyTuple *, PyDict *) -> PyResult<PyObject *> {
			return PyInteger::create(std::hash<std::thread::id>{}(std::this_thread::get_id()));
		}).unwrap());

	return s_thread_module;
}
}// namespace py
