#include "ExecutionFrame.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyType.hpp"
#include "runtime/types/builtin.hpp"


ExecutionFrame::ExecutionFrame() {}

ExecutionFrame *ExecutionFrame::create(ExecutionFrame *parent,
	size_t register_count,
	PyDict *globals,
	PyDict *locals
	/* PyDict *ns */)
{
	auto *new_frame = Heap::the().allocate<ExecutionFrame>();
	new_frame->m_parent = parent;
	new_frame->m_register_count = register_count;
	new_frame->m_globals = globals;
	new_frame->m_locals = locals;
	// new_frame->m_ns = ns;

	if (new_frame->m_parent) {
		new_frame->m_builtins = new_frame->m_parent->m_builtins;
	} else {
		ASSERT(new_frame->locals()->map().contains(String{ "__builtins__" }))
		ASSERT(std::get<PyObject *>((*new_frame->m_locals)[String{ "__builtins__" }])->type()
			   == module())
		// TODO: could this just return the builtin singleton?
		new_frame->m_builtins =
			as<PyModule>(std::get<PyObject *>((*new_frame->m_locals)[String{ "__builtins__" }]));
	}
	return new_frame;
}

void ExecutionFrame::set_exception_to_catch(PyObject *exception)
{
	m_exception_to_catch = exception;
}

void ExecutionFrame::set_exception(PyObject *exception) { m_exception = exception; }

bool ExecutionFrame::catch_exception(PyObject *exception) const
{
	if (m_exception_to_catch)
		return exception->type()->issubclass(m_exception_to_catch->type());
	else
		return false;
}

void ExecutionFrame::put_local(const std::string &name, PyObject *obj)
{
	m_locals->insert(String{ name }, obj);
}

void ExecutionFrame::put_global(const std::string &name, PyObject *obj)
{
	m_globals->insert(String{ name }, obj);
}


PyDict *ExecutionFrame::locals() const { return m_locals; }
PyDict *ExecutionFrame::globals() const { return m_globals; }
PyModule *ExecutionFrame::builtins() const { return m_builtins; }

ExecutionFrame *ExecutionFrame::exit()
{
	// if (m_ns) {
	// 	for (const auto &[k, v] : m_locals->map()) { m_ns->insert(k, v); }
	// }
	return m_parent;
}

std::string ExecutionFrame::to_string() const
{
	const auto locals = m_locals ? m_locals->to_string() : "";
	const auto globals = m_globals ? m_globals->to_string() : "";
	const auto builtins = m_builtins ? m_builtins->to_string() : "";
	// const auto ns = m_ns ? m_ns->to_string() : "";
	const void *parent = m_parent ? &m_parent : nullptr;

	return fmt::format(
		"ExecutionFrame(locals={}, globals={}, builtins={}, namespace={}, "
		"parent={})",
		locals,
		globals,
		builtins,
		0,// ns,
		parent);
}

void ExecutionFrame::visit_graph(Visitor &visitor)
{
	visitor.visit(*this);
	if (m_locals) m_locals->visit_graph(visitor);
	if (m_globals) m_globals->visit_graph(visitor);
	if (m_builtins) m_builtins->visit_graph(visitor);
	// if (m_ns) m_ns->visit_graph(visitor);
	for (const auto &val : m_parameters) {
		if (val.has_value() && std::holds_alternative<PyObject *>(*val)) {
			std::get<PyObject *>(*val)->visit_graph(visitor);
		}
	}
	if (m_parent) { m_parent->visit_graph(visitor); }
}
