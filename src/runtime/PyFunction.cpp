#include "PyFunction.hpp"
#include "PyBoundMethod.hpp"
#include "PyCell.hpp"
#include "PyCode.hpp"
#include "PyDict.hpp"
#include "PyFrame.hpp"
#include "PyModule.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "TypeError.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/Bytecode.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

#include "utilities.hpp"

namespace py {

PyFunction::PyFunction(PyType *type) : PyBaseObject(type) {}

PyFunction::PyFunction(std::string name,
	std::vector<Value> defaults,
	std::vector<Value> kwonly_defaults,
	PyCode *code,
	PyTuple *closure,
	PyDict *globals)
	: PyBaseObject(BuiltinTypes::the().function()), m_code(code), m_globals(globals),
	  m_defaults(std::move(defaults)), m_kwonly_defaults(std::move(kwonly_defaults)),
	  m_closure(closure)
{
	auto name_ = PyString::create(name);
	if (name_.is_err()) { TODO(); }
	m_name = name_.unwrap();

	// FIXME: get the docstring from PyCode
	auto doc_ = PyString::create("");
	if (doc_.is_err()) { TODO(); }
	m_doc = doc_.unwrap();

	auto dict_ = PyDict::create();
	if (dict_.is_err()) { TODO(); }
	m_dict = dict_.unwrap();
	m_attributes = m_dict;

	if (!m_closure) { m_closure = PyTuple::create().unwrap(); }

	m_dict->insert(String{ "__closure__" }, m_closure);
	m_dict->insert(String{ "__doc__" }, m_doc);
}

void PyFunction::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_code);
	if (m_globals) visitor.visit(*m_globals);
	if (m_module) visitor.visit(*m_module);
	if (m_dict) visitor.visit(*m_dict);
	if (m_name) visitor.visit(*m_name);
	if (m_doc) visitor.visit(*m_doc);
	if (m_closure) visitor.visit(*m_closure);
}

PyType *PyFunction::static_type() const { return function(); }

PyResult<PyObject *> PyFunction::__repr__() const
{
	return PyString::create(
		fmt::format("<function {} at {}>", m_name->value(), static_cast<const void *>(this)));
}

PyResult<PyObject *> PyFunction::__get__(PyObject *instance, PyObject * /*owner*/) const
{
	if (!instance || instance == py_none()) { return Ok(const_cast<PyFunction *>(this)); }
	return PyBoundMethod::create(instance, const_cast<PyFunction *>(this));
}

PyResult<PyObject *>
	PyFunction::call_with_frame(PyDict *locals, PyTuple *args, PyDict *kwargs) const
{
	return m_code->eval(m_globals,
		locals,
		args,
		kwargs,
		m_defaults,
		m_kwonly_defaults,
		m_closure ? m_closure->elements() : std::vector<Value>{},
		m_name);
}

PyResult<PyObject *> PyFunction::__call__(PyTuple *args, PyDict *kwargs)
{
	auto function_locals = PyDict::create();
	if (function_locals.is_err()) { return function_locals; }
	return call_with_frame(function_locals.unwrap(), args, kwargs);
}

std::string PyFunction::to_string() const
{
	return fmt::format("<function {} at {}>", m_name->value(), (void *)this);
}

namespace {
	std::once_flag function_flag;
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyFunction::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(function_flag, []() {
			type = std::move(klass<PyFunction>("function")
								 .attr("__code__", &PyFunction::m_code)
								 .attr("__globals__", &PyFunction::m_globals)
								 .attr("__dict__", &PyFunction::m_dict)
								 .attr("__name__", &PyFunction::m_name)
								 .attr("__doc__", &PyFunction::m_doc)
								 .attribute_readonly("__closure__", &PyFunction::m_closure)
								 .type);
		});
		return std::move(type);
	};
}

PyNativeFunction::PyNativeFunction(PyType *type) : PyBaseObject(type) {}

PyNativeFunction::PyNativeFunction(std::string &&name, FunctionType &&function)
	: PyBaseObject(BuiltinTypes::the().native_function()), m_name(std::move(name)),
	  m_function(std::move(function))
{}

std::string PyNativeFunction::to_string() const
{
	if (is_method()) {
		return fmt::format("<built-in method {} of {} object at {}>",
			m_name,
			m_self->type()->name(),
			(void *)this);
	} else {
		return fmt::format("<built-in function {} at {}>", m_name, (void *)this);
	}
}

PyResult<PyObject *> PyNativeFunction::__call__(PyTuple *args, PyDict *kwargs)
{
	if (is_method()) {
		ASSERT(m_self);
		return VirtualMachine::the().interpreter().call(this, m_self, args, kwargs);
	} else {
		return VirtualMachine::the().interpreter().call(this, args, kwargs);
	}
}

PyResult<PyObject *> PyNativeFunction::__repr__() const { return PyString::create(to_string()); }

void PyNativeFunction::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_self) { visitor.visit(*m_self); }
	for (auto *obj : m_captures) { visitor.visit(*obj); }
}

PyType *PyNativeFunction::static_type() const { return native_function(); }

namespace {
	std::once_flag native_function_flag;

	std::unique_ptr<TypePrototype> register_native_function()
	{
		return std::move(klass<PyNativeFunction>("builtin_function_or_method").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyNativeFunction::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(native_function_flag, []() { type = register_native_function(); });
		return std::move(type);
	};
}

template<> PyFunction *as(PyObject *node)
{
	if (node->type() == function()) { return static_cast<PyFunction *>(node); }
	return nullptr;
}

template<> const PyFunction *as(const PyObject *node)
{
	if (node->type() == function()) { return static_cast<const PyFunction *>(node); }
	return nullptr;
}

template<> PyNativeFunction *as(PyObject *node)
{
	if (node->type() == native_function()) { return static_cast<PyNativeFunction *>(node); }
	return nullptr;
}

template<> const PyNativeFunction *as(const PyObject *node)
{
	if (node->type() == native_function()) { return static_cast<const PyNativeFunction *>(node); }
	return nullptr;
}

}// namespace py
