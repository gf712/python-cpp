#include "PyFunction.hpp"
#include "PyDict.hpp"
#include "PyModule.hpp"
#include "executable/bytecode/Bytecode.hpp"

#include "utilities.hpp"

PyCode::PyCode(std::shared_ptr<Function> function,
	size_t function_id,
	std::vector<std::string> args,
	PyModule *module)
	: PyBaseObject(PyObjectType::PY_CODE), m_function(function), m_function_id(function_id),
	  m_register_count(function->registers_needed()), m_args(std::move(args)), m_module(module)
{}

size_t PyCode::register_count() const { return m_register_count; }

void PyCode::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	// FIXME: this should probably never be null
	if (m_module) m_module->visit_graph(visitor);
}

PyFunction::PyFunction(std::string name, PyCode *code, PyDict *globals)
	: PyBaseObject(PyObjectType::PY_FUNCTION), m_name(std::move(name)), m_code(code),
	  m_globals(globals)
{}


void PyFunction::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	m_code->visit_graph(visitor);
	// FIXME: this should probably never be null
	// FIXME: we shouldn't need to visit globals, since globals should be accessible by the parent
	//        object.
	//        The current issue with visiting globals is that it will create a cycle
	//        and the visitor currently doesn't handle cycles
	if (m_globals) visitor.visit(*m_globals);
}


void PyNativeFunction::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto *obj : m_captures) { obj->visit_graph(visitor); }
}