module;
#include "core.hpp"
#include "memory/allocate.hpp"
#include <cstdint>

module py.runtime;
import py.memory;
import std;

// After the import: these name module-owned types.
#include "PyWeakProxy.hpp"


namespace py {

namespace {
	static PyType *s_weak_proxy = nullptr;
}

PyWeakProxy::PyWeakProxy(PyType *type) : PyBaseObject(type) {}

PyWeakProxy::PyWeakProxy(PyObject *object, PyObject *callback)
	: PyBaseObject(s_weak_proxy), m_object(object), m_callback(callback)
{}

PyWeakProxy::~PyWeakProxy()
{
	if (m_object && m_object != py_none()) {
		VirtualMachine::the().heap().unregister_weakref(std::bit_cast<uint8_t *>(m_object), this);
	}
}

PyResult<PyWeakProxy *> PyWeakProxy::create(PyObject *object, PyObject *callback)
{
	auto *result = VirtualMachine::the().heap().allocate_weakref<PyWeakProxy>(object, callback);
	if (!result) { return Err(memory_error(sizeof(PyWeakProxy))); }
	return Ok(result);
}

void PyWeakProxy::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_callback) visitor.visit(*m_callback);
}

std::string PyWeakProxy::to_string() const
{
	if (!m_object) {
		return std::format("<weakproxy at {} empty", static_cast<const void *>(this));
	}
	return std::format("<weakproxy at {} to {} at {}>",
		static_cast<const void *>(this),
		m_object->type()->name(),
		static_cast<const void *>(m_object));
}

PyResult<PyObject *> PyWeakProxy::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_weak_proxy);
	ASSERT(args && args->size() > 0);
	ASSERT(!kwargs || kwargs->size() == 0);

	auto *obj = PyObject::from(args->elements()[0]).unwrap();
	auto *callback = [args]() -> PyObject * {
		if (args->size() > 1) { return PyObject::from(args->elements()[1]).unwrap(); }
		return nullptr;
	}();

	return PyWeakProxy::create(obj, callback);
}

PyResult<PyObject *> PyWeakProxy::__str__() const
{
	if (!m_object) {
		// FIXME: should be a ReferenceError
		return Err(value_error("weakly-referenced object no longer exists"));
	}
	return m_object->str();
}

PyResult<PyObject *> PyWeakProxy::__repr__() const
{
	if (!m_object) {
		// FIXME: should be a ReferenceError
		return Err(value_error("weakly-referenced object no longer exists"));
	}
	return PyString::create(to_string());
}

PyResult<PyObject *> PyWeakProxy::__getattribute__(PyObject *attribute) const
{
	auto obj = [this]() -> PyResult<const PyObject *> {
		if (type() == s_weak_proxy) {
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
		if (attribute->type() == s_weak_proxy) {
			if (!static_cast<const PyWeakProxy *>(attribute)->is_alive()) {
				// FIXME: should be a ReferenceError
				return Err(value_error("weakly-referenced object no longer exists"));
			} else {
				return Ok(static_cast<const PyWeakProxy *>(attribute)->m_object);
			}
		}
		return Ok(attribute);
	}();
	if (attr.is_err()) { return attr; }

	return obj.unwrap()->get_attribute(attr.unwrap());
}

PyType *PyWeakProxy::register_type(PyModule *module, std::string_view name)
{
	if (!s_weak_proxy) {
		s_weak_proxy = klass<PyWeakProxy>(module, "weakproxy")
						   .attr("__callback__", &PyWeakProxy::m_callback)
						   .finalize();
	}
	module->add_symbol(PyString::create(std::string(name)).unwrap(), s_weak_proxy);
	return s_weak_proxy;
}

bool PyWeakProxy::is_alive() const
{
	if (m_object
		&& !VirtualMachine::the().heap().has_weakref_object(std::bit_cast<uint8_t *>(m_object))) {
		m_object = py_none();
	}
	return m_object != py_none();
}

PyObject *PyWeakProxy::get_object() const { return m_object; }

}// namespace py
