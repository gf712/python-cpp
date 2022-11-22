#include "PyModule.hpp"
#include "MemoryError.hpp"
#include "PyDict.hpp"
#include "PyList.hpp"
#include "PyString.hpp"
#include "ValueError.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/codegen/BytecodeGenerator.hpp"
#include "interpreter/Interpreter.hpp"
#include "modules/Modules.hpp"
#include "parser/Parser.hpp"
#include "runtime/AttributeError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

#include <filesystem>

using namespace py;

PyModule::PyModule(PyDict *symbol_table, PyString *module_name, PyObject *doc)
	: PyBaseObject(BuiltinTypes::the().module()), m_module_name(module_name), m_doc(doc)
{
	m_package = PyString::create("").unwrap();
	m_loader = py_none();
	m_spec = py_none();

	m_attributes = symbol_table;
	m_dict = m_attributes;
	m_attributes->insert(String{ "__name__" }, m_module_name);
	m_attributes->insert(String{ "__doc__" }, m_doc);
	m_attributes->insert(String{ "__package__" }, m_package);
	m_attributes->insert(String{ "__loader__" }, m_loader);
	m_attributes->insert(String{ "__spec__" }, m_spec);
}

PyResult<PyObject *> PyModule::__repr__() const
{
	if (VirtualMachine::the().interpreter().importlib()) {
		auto module_repr = VirtualMachine::the().interpreter().importlib()->get_method(
			PyString::create("_module_repr").unwrap());
		return module_repr.and_then([this](PyObject *obj) {
			return obj->call(PyTuple::create(const_cast<PyModule *>(this)).unwrap(), nullptr);
		});
	} else {
		return PyString::create(fmt::format("<module {}>", m_module_name->value()));
	}
}

PyResult<PyObject *> PyModule::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == module());
	ASSERT(!kwargs || kwargs->map().empty());

	auto symbol_table = PyDict::create();
	if (symbol_table.is_err()) return symbol_table;

	auto *name = args->size() > 0 ? PyObject::from(args->elements()[0]).unwrap() : nullptr;
	auto *doc = args->size() > 1 ? PyObject::from(args->elements()[1]).unwrap() : py_none();
	if (!name) { TODO(); }
	if (!as<PyString>(name)) { TODO(); }

	return PyModule::create(symbol_table.unwrap(), as<PyString>(name), doc);
}

PyResult<int32_t> PyModule::__init__(PyTuple *args, PyDict *kwargs)
{
	ASSERT(args)
	ASSERT(!kwargs || kwargs->map().empty());

	auto *name = args->size() > 0 ? PyObject::from(args->elements()[0]).unwrap() : nullptr;
	auto *doc = args->size() > 1 ? PyObject::from(args->elements()[1]).unwrap() : py_none();
	if (!name) { TODO(); }
	if (!as<PyString>(name)) { TODO(); }

	m_module_name = as<PyString>(name);
	m_doc = doc;

	auto attr = PyDict::create();
	if (attr.is_err()) return Err(attr.unwrap_err());
	m_attributes = attr.unwrap();
	m_dict = m_attributes;

	m_attributes->insert(String{ "__name__" }, m_module_name);
	m_attributes->insert(String{ "__doc__" }, m_doc);

	return Ok(0);
}

void PyModule::add_symbol(PyString *key, const Value &value) { m_attributes->insert(key, value); }

namespace {
bool is_initializing(PyObject *spec)
{
	auto _initializing_str = PyString::create("_initializing");
	ASSERT(_initializing_str.is_ok())
	auto value = spec->get_attribute(_initializing_str.unwrap());
	if (value.is_err()) { return false; }
	auto is_true = truthy(value.unwrap(), VirtualMachine::the().interpreter());
	if (is_true.is_ok()) { return is_true.unwrap(); }
	return false;
}
}// namespace

PyResult<PyObject *> PyModule::__getattribute__(PyObject *attribute) const
{
	auto attr = PyObject::__getattribute__(attribute);
	if (attr.is_ok() || attr.unwrap_err()->type() != AttributeError::static_type()) { return attr; }

	String getattr_str{ "__getattr__" };
	String name_str{ "__name__" };

	if (auto it = m_attributes->map().find(getattr_str); it != m_attributes->map().end()) {
		auto getattr = PyObject::from(it->second);
		ASSERT(getattr.is_ok())
		auto args = PyTuple::create(attribute);
		ASSERT(args.is_ok())
		return getattr.unwrap()->call(args.unwrap(), nullptr);
	} else if (auto it = m_attributes->map().find(name_str); it != m_attributes->map().end()) {
		auto module_name = PyObject::from(it->second);
		ASSERT(module_name.is_ok())
		if (auto name = as<PyString>(module_name.unwrap())) {
			String spec_str{ "__spec__" };
			if (auto it = m_attributes->map().find(spec_str); it != m_attributes->map().end()) {
				auto spec = PyObject::from(it->second);
				ASSERT(spec.is_ok())
				if (is_initializing(spec.unwrap())) {
					return Err(
						attribute_error("partially initialized "
										"module '{}' has no attribute '{}' "
										"(most likely due to a circular import)",
							name->to_string(),
							attribute->to_string()));
				} else {
					return Err(attribute_error("module '{}' has no attribute '{}'",
						name->to_string(),
						attribute->to_string()));
				}
			}
		}
	}

	return Err(attribute_error("module has no attribute '{}'", attribute->to_string()));
}

void PyModule::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_module_name) visitor.visit(*m_module_name);
	if (m_doc) visitor.visit(*m_doc);
	if (m_package) visitor.visit(*m_package);
	if (m_loader) visitor.visit(*m_loader);
	if (m_spec) visitor.visit(*m_spec);
	if (m_dict) visitor.visit(*m_dict);
	if (m_program) { m_program->visit_functions(visitor); }
}

std::string PyModule::to_string() const
{
	return fmt::format("<module '{}'>", m_module_name->to_string());
}

PyResult<PyModule *> PyModule::create(PyDict *symbol_table, PyString *module_name, PyObject *doc)
{
	auto *result = VirtualMachine::the().heap().allocate<PyModule>(symbol_table, module_name, doc);
	if (!result) { return Err(memory_error(sizeof(PyModule))); }
	return Ok(result);
}

PyType *PyModule::type() const { return module(); }

void PyModule::set_program(std::shared_ptr<Program> program) { m_program = std::move(program); }

const std::shared_ptr<Program> &PyModule::program() const { return m_program; }

namespace {

std::once_flag module_flag;

std::unique_ptr<TypePrototype> register_module()
{
	return std::move(klass<PyModule>("module").attr("__dict__", &PyModule::m_dict).type);
}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyModule::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(module_flag, []() { type = ::register_module(); });
		return std::move(type);
	};
}

template<> PyModule *py::as(PyObject *obj)
{
	if (obj->type() == module()) { return static_cast<PyModule *>(obj); }
	return nullptr;
}


template<> const PyModule *py::as(const PyObject *obj)
{
	if (obj->type() == module()) { return static_cast<const PyModule *>(obj); }
	return nullptr;
}
