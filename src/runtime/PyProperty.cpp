#include "PyProperty.hpp"
#include "AttributeError.hpp"
#include "PyDict.hpp"
#include "PyFunction.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

PyResult<PyObject *> PyProperty::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::property());

	auto fget = [&]() -> PyResult<PyObject *> {
		if (args && args->size() >= 1) {
			return PyObject::from(args->elements()[0]);
		} else if (kwargs) {
			if (auto it = kwargs->map().find(String{ "fget" }); it != kwargs->map().end()) {
				return PyObject::from(it->second);
			}
		}
		return Ok(py_none());
	}();
	if (fget.is_err()) { return fget; }

	auto fset = [&]() -> PyResult<PyObject *> {
		if (args && args->size() >= 2) {
			return PyObject::from(args->elements()[1]);
		} else if (kwargs) {
			if (auto it = kwargs->map().find(String{ "fset" }); it != kwargs->map().end()) {
				return PyObject::from(it->second);
			}
		}
		return Ok(py_none());
	}();
	if (fset.is_err()) { return fset; }

	auto fdel = [&]() -> PyResult<PyObject *> {
		if (args && args->size() >= 3) {
			return PyObject::from(args->elements()[1]);
		} else if (kwargs) {
			if (auto it = kwargs->map().find(String{ "fdel" }); it != kwargs->map().end()) {
				return PyObject::from(it->second);
			}
		}
		return Ok(py_none());
	}();
	if (fdel.is_err()) { return fdel; }

	auto doc = [&]() -> PyResult<PyObject *> {
		if (args && args->size() >= 3) {
			return PyObject::from(args->elements()[1]);
		} else if (kwargs) {
			if (auto it = kwargs->map().find(String{ "doc" }); it != kwargs->map().end()) {
				return PyObject::from(it->second);
			}
		}
		return Ok(py_none());
	}();
	if (doc.is_err()) { return doc; }

	return PyProperty::create(fget.unwrap(), fset.unwrap(), fdel.unwrap(), doc.unwrap());
}

PyProperty::PyProperty(PyType *type) : PyBaseObject(type) {}

PyProperty::PyProperty(PyObject *fget, PyObject *fset, PyObject *fdel, PyObject *name)
	: PyBaseObject(types::BuiltinTypes::the().property()), m_getter(fget), m_setter(fset),
	  m_deleter(fdel), m_property_name(name)
{}

std::string PyProperty::to_string() const
{
	return fmt::format("<property object at {}>", static_cast<const void *>(this));
}

PyResult<PyProperty *>
	PyProperty::create(PyObject *fget, PyObject *fset, PyObject *fdel, PyObject *name)
{
	auto *obj = VirtualMachine::the().heap().allocate<PyProperty>(fget, fset, fdel, name);
	if (!obj) return Err(memory_error(sizeof(PyProperty)));
	return Ok(obj);
}

PyResult<PyObject *> PyProperty::getter(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!kwargs || kwargs->map().empty());
	ASSERT(args);
	ASSERT(args->size() == 1);

	auto getter_ = PyObject::from(args->elements()[0]);

	if (getter_.is_err()) return getter_;

	return PyProperty::create(getter_.unwrap(),
		m_setter == py_none() ? py_none() : m_setter,
		m_deleter == py_none() ? py_none() : m_deleter,
		m_property_name);
}

PyResult<PyObject *> PyProperty::setter(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!kwargs || kwargs->map().empty());
	ASSERT(args);
	ASSERT(args->size() == 1);

	auto setter_ = PyObject::from(args->elements()[0]);

	if (setter_.is_err()) return setter_;

	return PyProperty::create(m_getter == py_none() ? py_none() : m_getter,
		setter_.unwrap(),
		m_deleter == py_none() ? py_none() : m_deleter,
		m_property_name);
}

PyResult<PyObject *> PyProperty::deleter(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!kwargs || kwargs->map().empty());
	ASSERT(args);
	ASSERT(args->size() == 1);

	auto deleter_ = PyObject::from(args->elements()[0]);

	if (deleter_.is_err()) return deleter_;

	return PyProperty::create(m_setter == py_none() ? py_none() : m_setter,
		m_setter == py_none() ? py_none() : m_setter,
		deleter_.unwrap(),
		m_property_name);
}

PyResult<PyObject *> PyProperty::__repr__() const
{
	return PyString::create(PyProperty::to_string());
}

PyResult<PyObject *> PyProperty::__get__(PyObject *instance, PyObject *) const
{
	if (!instance || instance == py_none()) { return Ok(const_cast<PyProperty *>(this)); }

	if (!m_getter) {
		if (m_property_name) {
			return Err(attribute_error("unreadable attribute {}", m_property_name->to_string()));
		} else {
			return Err(attribute_error("unreadable attribute"));
		}
	}

	return PyTuple::create(instance).and_then(
		[&](auto *args) { return m_getter->call(args, nullptr); });
}

PyResult<std::monostate> PyProperty::__set__(PyObject *obj, PyObject *value)
{
	auto func = [this, value] {
		if (value == nullptr) {
			return m_deleter;
		} else {
			return m_setter;
		}
	}();

	if (!func) {
		if (!value) {
			return Err(attribute_error("can't delete attribute"));
		} else {
			return Err(attribute_error("can't set attribute"));
		}
	}

	auto args = PyTuple::create(obj, value);
	if (args.is_err()) return Err(args.unwrap_err());

	return func->call(args.unwrap(), nullptr).and_then([](auto *) { return Ok(std::monostate{}); });
}

void PyProperty::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_getter) { visitor.visit(*m_getter); }
	if (m_setter) { visitor.visit(*m_setter); }
	if (m_deleter) { visitor.visit(*m_deleter); }
	if (m_property_name) { visitor.visit(*m_property_name); }
}

PyType *PyProperty::static_type() const { return types::property(); }

namespace {

	std::once_flag property_flag;

	std::unique_ptr<TypePrototype> register_property()
	{
		return std::move(klass<PyProperty>("property")
				.def("getter", &PyProperty::getter)
				.def("deleter", &PyProperty::deleter)
				.def("setter", &PyProperty::setter)
				.attr("fget", &PyProperty::m_getter)
				.attr("fset", &PyProperty::m_setter)
				.attr("fdel", &PyProperty::m_deleter)
				.type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyProperty::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(property_flag, []() { type = register_property(); });
		return std::move(type);
	};
}

template<> PyProperty *as(PyObject *obj)
{
	if (obj->type() == types::property()) { return static_cast<PyProperty *>(obj); }
	return nullptr;
}

template<> const PyProperty *as(const PyObject *obj)
{
	if (obj->type() == types::property()) { return static_cast<const PyProperty *>(obj); }
	return nullptr;
}

}// namespace py
