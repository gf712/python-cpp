#include "PyNamespace.hpp"
#include "MemoryError.hpp"
#include "NotImplemented.hpp"
#include "PyDict.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyNamespace *as(PyObject *obj)
{
	if (obj->type() == types::namespace_()) { return static_cast<PyNamespace *>(obj); }
	return nullptr;
}


template<> const PyNamespace *as(const PyObject *obj)
{
	if (obj->type() == types::namespace_()) { return static_cast<const PyNamespace *>(obj); }
	return nullptr;
}


PyNamespace::PyNamespace(PyType *type) : PyBaseObject(type) {}

PyNamespace::PyNamespace(PyDict *dict)
	: PyBaseObject(types::BuiltinTypes::the().namespace_()), m_dict(dict)
{
	m_attributes = m_dict;
}

PyResult<PyNamespace *> PyNamespace::create()
{
	auto dict = PyDict::create();
	if (dict.is_err()) return Err(dict.unwrap_err());
	return create(dict.unwrap());
}

PyResult<PyNamespace *> PyNamespace::create(PyDict *dict)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyNamespace>(dict)) { return Ok(obj); }
	return Err(memory_error(sizeof(PyNamespace)));
}

PyResult<PyObject *> PyNamespace::__new__(const PyType *type, PyTuple *, PyDict *)
{
	ASSERT(type == types::namespace_());
	return PyNamespace::create();
}

PyResult<int32_t> PyNamespace::__init__(PyTuple *args, PyDict *kwargs)
{
	if (args || args->size() > 0) { return Err(type_error("no positional arguments expected")); }
	if (kwargs) {
		for (const auto &[key, value] : kwargs->map()) {
			if (std::holds_alternative<PyObject *>(key)) {
				if (!as<PyString>(std::get<PyObject *>(key))) {
					return Err(type_error("keywords must be strings"));
				}
			} else if (!std::holds_alternative<String>(key)) {
				return Err(type_error("keywords must be strings"));
			}
		}
		auto result = m_dict->merge(PyTuple::create(kwargs).unwrap(), nullptr);
		if (result.is_err()) return Err(result.unwrap_err());
	}
	return Ok(0);
}

PyResult<PyObject *> PyNamespace::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyNamespace::__eq__(const PyObject *obj) const
{
	if (!as<PyNamespace>(obj)) { return Ok(not_implemented()); }
	return m_dict->richcompare(as<PyNamespace>(obj), RichCompare::Py_EQ);
}

PyResult<PyObject *> PyNamespace::__lt__(const PyObject *obj) const
{
	if (!as<PyNamespace>(obj)) { return Ok(not_implemented()); }
	return m_dict->richcompare(as<PyNamespace>(obj), RichCompare::Py_LT);
}

void PyNamespace::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_dict) visitor.visit(*m_dict);
}

namespace {

	std::once_flag namespace_flag;

	std::unique_ptr<TypePrototype> register_namespace()
	{
		return std::move(klass<PyNamespace>("types.SimpleNamespace")
							 .attribute_readonly("__dict__", &PyNamespace::m_dict)
							 .type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyNamespace::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(namespace_flag, []() { type = register_namespace(); });
		return std::move(type);
	};
}

PyType *PyNamespace::static_type() const { return types::namespace_(); }

std::string PyNamespace::to_string() const
{
	std::stringstream ss;
	for (size_t i = 0; const auto &[key, value] : m_dict->map()) {
		ss << PyObject::from(key).unwrap()->to_string() << "="
		   << PyObject::from(value).unwrap()->to_string();
		i++;
		if (i != m_dict->map().size()) { ss << ", "; }
	}
	return fmt::format("namespace({})", ss.str());
}


}// namespace py
