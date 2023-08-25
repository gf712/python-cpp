#include "PyGenericAlias.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "RuntimeError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyGenericAlias *as(PyObject *obj)
{
	if (obj->type() == types::generic_alias()) { return static_cast<PyGenericAlias *>(obj); }
	return nullptr;
}

template<> const PyGenericAlias *as(const PyObject *obj)
{
	if (obj->type() == types::generic_alias()) { return static_cast<const PyGenericAlias *>(obj); }
	return nullptr;
}

PyGenericAlias::PyGenericAlias(PyType *type) : PyBaseObject(type) {}

PyGenericAlias::PyGenericAlias(PyObject *origin, PyTuple *args, PyObject *parameters)
	: PyBaseObject(types::BuiltinTypes::the().generic_alias()), m_origin(origin), m_args(args),
	  m_parameters(parameters)
{}

PyResult<PyGenericAlias *>
	PyGenericAlias::create(PyObject *origin, PyObject *args, PyObject *parameters)
{
	auto &heap = VirtualMachine::the().heap();
	auto *args_as_tuple = as<PyTuple>(args);
	if (!args_as_tuple) {
		auto args_ = PyTuple::create(args);
		if (args_.is_err()) return Err(args_.unwrap_err());
		args_as_tuple = args_.unwrap();
	}
	if (auto *obj = heap.allocate<PyGenericAlias>(origin, args_as_tuple, parameters)) {
		return Ok(obj);
	}
	return Err(memory_error(sizeof(PyGenericAlias)));
}

PyResult<PyGenericAlias *> PyGenericAlias::create(PyObject *origin, PyObject *args)
{
	return PyGenericAlias::create(origin, args, nullptr);
}


std::string PyGenericAlias::to_string() const
{
	return fmt::format("<generic_alias object at {}>", static_cast<const void *>(this));
}

PyResult<PyObject *> PyGenericAlias::__repr__() const
{
	// FIXME: this is a very simplified (and incorrect) version of the actual way generic_alias
	//        should be represented
	if (!m_origin) { return Err(runtime_error("generic_alias origin not set")); }
	std::ostringstream os;
	for (size_t i = 0; const auto &el : m_args->elements()) {
		auto type_ = PyObject::from(el);
		ASSERT(type_.is_ok());
		ASSERT(as<PyType>(type_.unwrap()));
		os << as<PyType>(type_.unwrap())->name();
		if (++i != m_args->elements().size()) { os << ", "; }
	}
	ASSERT(as<PyType>(m_origin));
	return PyString::create(fmt::format("{}[{}]", as<PyType>(m_origin)->name(), os.str()));
}

namespace {
	std::once_flag generic_alias_flag;

	std::unique_ptr<TypePrototype> register_generic_alias()
	{
		return std::move(klass<PyGenericAlias>("types.GenericAlias").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyGenericAlias::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(generic_alias_flag, []() { type = register_generic_alias(); });
		return std::move(type);
	};
}

PyType *PyGenericAlias::static_type() const { return types::generic_alias(); }

void PyGenericAlias::visit_graph(Visitor &visitor)
{
	if (m_origin) visitor.visit(*m_origin);
	if (m_args) visitor.visit(*m_args);
	if (m_parameters) visitor.visit(*m_parameters);
}

}// namespace py
