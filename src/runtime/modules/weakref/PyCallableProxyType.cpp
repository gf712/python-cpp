#include "PyCallableProxyType.hpp"
#include "runtime/MemoryError.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyType.hpp"
#include "runtime/ValueError.hpp"
#include "runtime/types/api.hpp"
#include "vm/VM.hpp"

namespace py {

namespace {
	static PyType *s_weak_callableproxy = nullptr;
}

PyCallableProxyType::PyCallableProxyType(PyType *type) : PyBaseObject(type) {}

PyCallableProxyType::PyCallableProxyType(PyObject *object, PyObject *callback)
	: PyBaseObject(s_weak_callableproxy), m_object(object), m_callback(callback)
{}

PyResult<PyCallableProxyType *> PyCallableProxyType::create(PyObject *object, PyObject *callback)
{
	auto *result =
		VirtualMachine::the().heap().allocate_weakref<PyCallableProxyType>(object, callback);
	if (!result) { return Err(memory_error(sizeof(PyCallableProxyType))); }
	return Ok(result);
}

void PyCallableProxyType::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_callback) visitor.visit(*m_callback);
}

std::string PyCallableProxyType::to_string() const
{
	if (!m_object) {
		return fmt::format("<weakproxy at {} empty", static_cast<const void *>(this));
	}
	return fmt::format("<weakproxy at {} to {} at {}>",
		static_cast<const void *>(this),
		m_object->type()->name(),
		static_cast<const void *>(m_object));
}

PyResult<PyObject *> PyCallableProxyType::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_weak_callableproxy);
	ASSERT(args && args->size() > 0);
	ASSERT(!kwargs || kwargs->size() == 0);

	auto *obj = PyObject::from(args->elements()[0]).unwrap();
	auto *callback = [args]() -> PyObject * {
		if (args->size() > 1) { return PyObject::from(args->elements()[1]).unwrap(); }
		return nullptr;
	}();

	return PyCallableProxyType::create(obj, callback);
}

PyResult<PyObject *> PyCallableProxyType::__str__() const
{
	if (!m_object) {
		// FIXME: should be a ReferenceError
		return Err(value_error("weakly-referenced object no longer exists"));
	}
	return m_object->str();
}

PyResult<PyObject *> PyCallableProxyType::__repr__() const
{
	if (!m_object) {
		// FIXME: should be a ReferenceError
		return Err(value_error("weakly-referenced object no longer exists"));
	}
	return PyString::create(to_string());
}

PyResult<PyObject *> PyCallableProxyType::__getattribute__(PyObject *attribute) const
{
	auto obj = [this]() -> PyResult<const PyObject *> {
		if (type() == s_weak_callableproxy) {
			if (!is_alive()) {
				// FIXME: should be a ReferenceError
				return Err(value_error("weakly-referenced object no longer exists"));
			} else {
				return Ok(m_object);
			}
		}
		return Ok(this);
	}();
	if (obj.is_err()) { return Err(obj.unwrap_err()); }

	auto attr = [attribute]() -> PyResult<PyObject *> {
		if (attribute->type() == s_weak_callableproxy) {
			if (!static_cast<const PyCallableProxyType *>(attribute)->is_alive()) {
				// FIXME: should be a ReferenceError
				return Err(value_error("weakly-referenced object no longer exists"));
			} else {
				return Ok(static_cast<const PyCallableProxyType *>(attribute)->m_object);
			}
		}
		return Ok(attribute);
	}();
	if (attr.is_err()) { return attr; }

	return obj.unwrap()->get_attribute(attr.unwrap());
}

PyResult<PyObject *> PyCallableProxyType::__call__(PyTuple *args, PyDict *kwargs)
{
	if (!m_object) {
		// FIXME: should be a ReferenceError
		return Err(value_error("weakly-referenced object no longer exists"));
	}
	return m_object->call(args, kwargs);
}

PyType *PyCallableProxyType::register_type(PyModule *module, std::string_view name)
{
	if (!s_weak_callableproxy) {
		s_weak_callableproxy = klass<PyCallableProxyType>(module, "weakcallableproxy")
								   .attr("__callback__", &PyCallableProxyType::m_callback)
								   .finalize();
	}
	module->add_symbol(PyString::create(std::string(name)).unwrap(), s_weak_callableproxy);
	return s_weak_callableproxy;
}

bool PyCallableProxyType::is_alive() const
{
	if (m_object
		&& !VirtualMachine::the().heap().has_weakref_object(bit_cast<uint8_t *>(m_object))) {
		m_object = nullptr;
	}
	return m_object != nullptr;
}

}// namespace py
