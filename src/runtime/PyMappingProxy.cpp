#include "PyMappingProxy.hpp"
#include "MemoryError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {
PyMappingProxy::PyMappingProxy(PyObject *mapping)
	: PyBaseObject(BuiltinTypes::the().mappingproxy()), m_mapping(mapping)
{}

PyResult<PyObject *> PyMappingProxy::create(PyObject *mapping)
{
	if (as<PyList>(mapping) || as<PyTuple>(mapping) || mapping->as_mapping().is_ok()) {
		auto *result = VirtualMachine::the().heap().allocate<PyMappingProxy>(mapping);
		if (!result) { return Err(memory_error(sizeof(PyMappingProxy))); }
		return Ok(result);
	}
	return Err(
		type_error("mappingproxy() argument must be a mapping, not '{}'", mapping->type()->name()));
}

std::string PyMappingProxy::to_string() const { return "mappingproxy"; }

PyResult<PyObject *> PyMappingProxy::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == mappingproxy());
	ASSERT(args && args->size() == 1);
	ASSERT(!kwargs || kwargs->map().empty());
	return PyObject::from(args->elements()[0]).and_then([](PyObject *mapping) {
		return PyMappingProxy::create(mapping);
	});
}

PyResult<PyObject *> PyMappingProxy::__repr__() const
{
	if (auto r = m_mapping->repr(); r.is_ok()) {
		return PyString::create(fmt::format("mappingproxy({})", r.unwrap()->to_string()));
	} else {
		return r;
	}
}

PyResult<PyObject *> PyMappingProxy::__getitem__(PyObject *index)
{
	return m_mapping->as_mapping().and_then(
		[index](PyMappingWrapper mapping_wrapper) -> PyResult<PyObject *> {
			return mapping_wrapper.getitem(index);
		});
}

PyType *PyMappingProxy::type() const { return mappingproxy(); }

void PyMappingProxy::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_mapping) { visitor.visit(*m_mapping); }
}

namespace {

	std::once_flag mappingproxy_flag;

	std::unique_ptr<TypePrototype> mappingproxy_reversed()
	{
		return std::move(klass<PyMappingProxy>("mappingproxy").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyMappingProxy::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(mappingproxy_flag, []() { type = mappingproxy_reversed(); });
		return std::move(type);
	};
}

}// namespace py
