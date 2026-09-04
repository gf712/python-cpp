module;
#include "core.hpp"
#include "memory/allocate.hpp"

module py.runtime;
import py.memory;
import py.types;

// These name py:: types, so they belong in the purview where the implicit
// import of py.runtime has already happened.
#include "PyCallableProxyType.hpp"
#include "PyWeakProxy.hpp"
#include "PyWeakRef.hpp"


namespace py {

namespace {
	PyType *weakref_ref = nullptr;
	PyType *weakref_proxy = nullptr;
}// namespace

PyResult<PyObject *> getweakrefcount(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
		kwargs,
		"getweakrefcount",
		std::integral_constant<std::size_t, 1>{},
		std::integral_constant<std::size_t, 1>{});

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [object] = result.unwrap();

	return PyInteger::create(
		VirtualMachine::the().heap().weakref_count(std::bit_cast<std::uint8_t *>(object)));
}

PyResult<PyObject *> getweakrefs(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
		kwargs,
		"getweakrefs",
		std::integral_constant<std::size_t, 1>{},
		std::integral_constant<std::size_t, 1>{});

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [object] = result.unwrap();

	auto weakrefs =
		VirtualMachine::the().heap().get_weakrefs(std::bit_cast<std::uint8_t *>(object));

	auto weakref_list = PyList::create();
	if (result.is_err()) { return weakref_list; }
	// Heap::get_weakrefs returns Cell* so that memory/ does not have to name
	// py::PyObject (that dependency closed a module cycle); every weakref stored
	// there is a PyObject, so the downcast is safe.
	auto &elements = weakref_list.unwrap()->elements();
	elements.reserve(elements.size() + weakrefs.size());
	for (auto *cell : weakrefs) { elements.push_back(static_cast<PyObject *>(cell)); }

	return weakref_list;
}

PyResult<PyObject *> proxy(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyObject *, PyObject *>::unpack_tuple(args,
		kwargs,
		"proxy",
		std::integral_constant<std::size_t, 1>{},
		std::integral_constant<std::size_t, 2>{},
		nullptr);

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [object, callback] = result.unwrap();

	if (object->is_callable()) { return PyCallableProxyType::create(object, callback); }

	return PyWeakProxy::create(object, callback);
}

PyResult<PyObject *> remove_dead_weakref(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyObject *, PyObject *>::unpack_tuple(args,
		kwargs,
		"remove_dead_weakref",
		std::integral_constant<std::size_t, 2>{},
		std::integral_constant<std::size_t, 2>{});

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [dict, key] = result.unwrap();

	if (!dict->type()->issubclass(types::dict())) { return Err(value_error("internal error")); }

	auto &d = static_cast<PyDict &>(*dict);
	auto it = d.map().find(key);
	if (it != d.map().end()) {
		auto value = PyObject::from(it->second);
		if (value.is_err()) { return value; }
		if (value.unwrap()->type()->issubclass(weakref_ref)
			&& static_cast<const PyWeakRef &>(*value.unwrap()).get_object() == py_none()) {
			d.remove(value.unwrap());
		} else if (value.unwrap()->type()->issubclass(weakref_proxy)
				   && static_cast<const PyWeakProxy &>(*value.unwrap()).get_object() == py_none()) {
			d.remove(value.unwrap());
		} else {
			return Err(type_error("not a weakref"));
		}
	}

	return Ok(py_none());
}

PyModule *weakref_module()
{
	auto *s_weak_ref_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("_weakref").unwrap(),
		PyString::create("").unwrap())
								  .unwrap();

	weakref_ref = PyWeakRef::register_type(s_weak_ref_module, "ref");
	PyWeakRef::register_type(s_weak_ref_module, "ReferenceType");
	weakref_proxy = PyWeakProxy::register_type(s_weak_ref_module, "ProxyType");
	PyCallableProxyType::register_type(s_weak_ref_module, "CallableProxyType");

	s_weak_ref_module->add_symbol(
		PyString::create("proxy").unwrap(), PyNativeFunction::create("proxy", &proxy).unwrap());

	s_weak_ref_module->add_symbol(PyString::create("getweakrefcount").unwrap(),
		PyNativeFunction::create("getweakrefcount", &getweakrefcount).unwrap());

	s_weak_ref_module->add_symbol(PyString::create("getweakrefs").unwrap(),
		PyNativeFunction::create("getweakrefs", &getweakrefs).unwrap());

	s_weak_ref_module->add_symbol(PyString::create("_remove_dead_weakref").unwrap(),
		PyNativeFunction::create("_remove_dead_weakref", &remove_dead_weakref).unwrap());

	return s_weak_ref_module;
}
}// namespace py
