#include "PyWeakRef.hpp"
#include "MemoryError.hpp"
#include "PyModule.hpp"
#include "PyNone.hpp"
#include "PyType.hpp"
#include "types/api.hpp"
#include "vm/VM.hpp"

namespace py {

namespace {
	static PyType *s_weak_ref = nullptr;
}

PyWeakRef::PyWeakRef(PyType *type) : PyBaseObject(type) {}

PyWeakRef::PyWeakRef(PyObject *object, PyObject *callback)
	: PyBaseObject(s_weak_ref), m_object(object), m_callback(callback)
{}

PyResult<PyWeakRef *> PyWeakRef::create(PyObject *object, PyObject *callback)
{
	auto *result = VirtualMachine::the().heap().allocate<PyWeakRef>(object, callback);
	if (!result) { return Err(memory_error(sizeof(PyWeakRef))); }
	return Ok(result);
}

void PyWeakRef::visit_graph(Visitor &visitor)
{
	// FIXME: should not keep strong reference to object
	if (m_object) visitor.visit(*m_object);
	if (m_callback) visitor.visit(*m_callback);
}

std::string PyWeakRef::to_string() const
{
	return fmt::format("<weakref at {}; {}>",
		static_cast<const void *>(this),
		is_alive()
			? fmt::format("to '{}' at {}", m_object->type()->name(), static_cast<void *>(m_object))
			: "dead");
}

PyResult<PyObject *> PyWeakRef::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == s_weak_ref)
	ASSERT(args && args->size() > 0)
	ASSERT(!kwargs || kwargs->size() == 0)

	auto *obj = PyObject::from(args->elements()[0]).unwrap();
	auto *callback = [args]() -> PyObject * {
		if (args->size() > 1) { return PyObject::from(args->elements()[1]).unwrap(); }
		return nullptr;
	}();

	return PyWeakRef::create(obj, callback);
}

PyResult<PyObject *> PyWeakRef::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyWeakRef::__call__(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!args || args->size() == 0)
	ASSERT(!kwargs || kwargs->size() == 0)
	if (is_alive()) {
		ASSERT(m_object)
		return Ok(m_object);
	} else {
		return Ok(py_none());
	}
}

PyType *PyWeakRef::register_type(PyModule *module, std::string_view name)
{
	if (!s_weak_ref) {
		s_weak_ref = klass<PyWeakRef>(module, "weakref")
						 .attr("__callback__", &PyWeakRef::m_callback)
						 .finalize();
	}
	module->add_symbol(PyString::create(std::string(name)).unwrap(), s_weak_ref);
	return s_weak_ref;
}

bool PyWeakRef::is_alive() const { return true; }

}// namespace py
