#include "ExecutionFrame.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyObject.hpp"


ExecutionFrame::ExecutionFrame() {}


std::shared_ptr<ExecutionFrame> ExecutionFrame::create(std::shared_ptr<ExecutionFrame> parent,
	PyDict *globals,
	PyDict *locals,
	PyDict *ns)
{
	auto new_frame = std::shared_ptr<ExecutionFrame>(new ExecutionFrame{});
	new_frame->m_parent = parent;
	new_frame->m_globals = globals;
	new_frame->m_locals = locals;
	new_frame->m_ns = ns;

	if (new_frame->m_parent) {
		new_frame->m_builtins = new_frame->m_parent->m_builtins;
	} else {
		// check return is not None?
		// assert that this is indeed a heap allocated PyModule
		new_frame->m_builtins =
			as<PyModule>(std::get<PyObject *>((*new_frame->m_locals)[String{ "__builtins__" }]));
	}
	return new_frame;
}

void ExecutionFrame::set_exception_to_catch(PyObject *exception)
{
	m_exception_to_catch = std::move(exception);
}

void ExecutionFrame::set_exception(PyObject *exception) { m_exception = std::move(exception); }

bool ExecutionFrame::catch_exception(PyObject *exception) const
{
	if (m_exception_to_catch)
		return m_exception_to_catch == exception;
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

std::shared_ptr<ExecutionFrame> ExecutionFrame::exit()
{
	if (m_ns) {
		for (const auto &[k, v] : m_locals->map()) { m_ns->insert(k, v); }
	}
	return m_parent;
}
