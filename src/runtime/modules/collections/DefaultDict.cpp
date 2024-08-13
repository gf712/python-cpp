#include "DefaultDict.hpp"
#include "runtime/KeyError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/Value.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"
#include <variant>

using namespace py;
using namespace py::collections;

namespace {
PyType *s_collections_defaultdict = nullptr;
}

DefaultDict::DefaultDict(PyType *type) : PyDict(type) {}

PyResult<int32_t> DefaultDict::__init__(PyTuple *args, PyDict *kwargs)
{
	if (args->elements().size() > 0) {
		auto default_factory = PyObject::from(args->elements()[0]);
		if (default_factory.is_err()) { return Err(default_factory.unwrap_err()); }
		m_default_factory = default_factory.unwrap();

		auto newargs = PyTuple::create(
			std::vector<Value>{ args->elements().begin() + 1, args->elements().end() });
		if (newargs.is_err()) { return Err(newargs.unwrap_err()); }
		args = newargs.unwrap();
	}

	return PyDict::__init__(args, kwargs);
}

PyResult<PyObject *> DefaultDict::__missing__(PyObject *key)
{
	if (!m_default_factory || m_default_factory == py_none()) {
		return KeyError::create(PyTuple::create(key).unwrap());
	}
	auto default_value = m_default_factory->call(nullptr, nullptr);
	if (default_value.is_err()) { return default_value; }

	return setitem(key, default_value.unwrap())
		.and_then([](auto) -> PyResult<PyObject *> { return Ok(py_none()); })
		.and_then([default_value](auto) -> PyResult<PyObject *> { return default_value; });
}

void DefaultDict::visit_graph(Visitor &visitor)
{
	PyDict::visit_graph(visitor);
	if (m_default_factory) { visitor.visit(*m_default_factory); }
}

PyType *DefaultDict::register_type(PyModule *module)
{
	if (!s_collections_defaultdict) {
		s_collections_defaultdict =
			klass<DefaultDict>(module, "collections.defaultdict", types::dict())
				.def("__missing__", &DefaultDict::__missing__)
				.finalize();
	}
	return s_collections_defaultdict;
}