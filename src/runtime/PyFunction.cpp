#include "PyFunction.hpp"
#include "PyDict.hpp"
#include "PyModule.hpp"
#include "executable/bytecode/Bytecode.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

#include "utilities.hpp"

PyCode::PyCode(std::shared_ptr<Function> function,
	size_t function_id,
	std::vector<std::string> args,
	PyModule *module)
	: PyBaseObject(PyObjectType::PY_CODE, BuiltinTypes::the().code()), m_function(function), m_function_id(function_id),
	  m_register_count(function->registers_needed()), m_args(std::move(args)), m_module(module)
{}

size_t PyCode::register_count() const { return m_register_count; }

void PyCode::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	// FIXME: this should probably never be null
	if (m_module) m_module->visit_graph(visitor);
}

PyType *PyCode::type_() const { return code(); }

namespace {

std::once_flag code_flag;

std::unique_ptr<TypePrototype> register_code() { return std::move(klass<PyCode>("code").type); }
}// namespace

std::unique_ptr<TypePrototype> PyCode::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(code_flag, []() { type = ::register_code(); });
	return std::move(type);
}

PyFunction::PyFunction(std::string name, PyCode *code, PyDict *globals)
	: PyBaseObject(PyObjectType::PY_FUNCTION, BuiltinTypes::the().function()), m_name(std::move(name)), m_code(code),
	  m_globals(globals)
{}

void PyFunction::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	m_code->visit_graph(visitor);
	if (m_globals) visitor.visit(*m_globals);
}

PyType *PyFunction::type_() const { return function(); }

namespace {

std::once_flag function_flag;

std::unique_ptr<TypePrototype> register_function()
{
	return std::move(klass<PyFunction>("function").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyFunction::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(function_flag, []() { type = ::register_function(); });
	return std::move(type);
}


PyNativeFunction::PyNativeFunction(std::string name,
	std::function<PyObject *(PyTuple *, PyDict *)> function)
	: PyBaseObject(PyObjectType::PY_NATIVE_FUNCTION, BuiltinTypes::the().native_function()), m_name(std::move(name)),
	  m_function(std::move(function))
{}

void PyNativeFunction::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto *obj : m_captures) { obj->visit_graph(visitor); }
}

PyType *PyNativeFunction::type_() const { return native_function(); }

namespace {

std::once_flag native_function_flag;

std::unique_ptr<TypePrototype> register_native_function()
{
	return std::move(klass<PyNativeFunction>("native_function").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyNativeFunction::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(native_function_flag, []() { type = ::register_native_function(); });
	return std::move(type);
}