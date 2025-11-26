#include "../Modules.hpp"
#include "Lock.hpp"
#include "RLock.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/types/builtin.hpp"

#include <thread>

namespace py {

PyResult<PyObject *> start_new_thread(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyObject *, PyObject *>::unpack_tuple(args,
		kwargs,
		"start_new_thread",
		std::integral_constant<size_t, 2>{},
		std::integral_constant<size_t, 2>{});

	if (result.is_err()) { return Ok(result.unwrap_err()); }

	auto [fn, fn_args] = result.unwrap();

	if (!fn_args->type()->issubclass(types::tuple())) {
		return Err(type_error("expected tuple but got {}", fn_args->type()->name()));
	}

	return fn->call(static_cast<PyTuple *>(fn_args), nullptr);
}

PyResult<PyObject *> _set_sentinel(PyTuple *, PyDict *) { return Lock::create(); }

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

	s_thread_module->add_symbol(PyString::create("start_new_thread").unwrap(),
		PyNativeFunction::create("start_new_thread", start_new_thread).unwrap());

	s_thread_module->add_symbol(PyString::create("_set_sentinel").unwrap(),
		PyNativeFunction::create("_set_sentinel", _set_sentinel).unwrap());

	s_thread_module->add_symbol(PyString::create("error").unwrap(), types::runtime_error());

	s_thread_module->add_symbol(
		PyString::create("TIMEOUT_MAX").unwrap(), PyInteger::create(LLONG_MAX).unwrap());

	pthread_attr_t attr;
	size_t stacksize;
	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &stacksize);

	s_thread_module->add_symbol(
		PyString::create("stack_size").unwrap(), PyInteger::create(stacksize).unwrap());

	return s_thread_module;
}
}// namespace py
