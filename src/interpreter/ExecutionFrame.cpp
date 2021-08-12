#include "ExecutionFrame.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyObject.hpp"


ExecutionFrame::ExecutionFrame() {}


std::shared_ptr<ExecutionFrame> ExecutionFrame::create(std::shared_ptr<ExecutionFrame> parent,
	std::shared_ptr<PyDict> globals,
	std::unique_ptr<PyDict> &&locals,
	std::shared_ptr<PyDict> ns)
{
	auto new_frame = std::shared_ptr<ExecutionFrame>(new ExecutionFrame{});
	new_frame->m_parent = std::move(parent);
	new_frame->m_globals = std::move(globals);
	new_frame->m_locals = std::move(locals);
	new_frame->m_ns = std::move(ns);

	if (new_frame->m_parent) {
		new_frame->m_builtins = new_frame->m_parent->m_builtins;
	} else {
		// check return is not None?
		// assert that this is indeed a heap allocated PyModule
		new_frame->m_builtins = as<PyModule>(
			std::get<std::shared_ptr<PyObject>>((*new_frame->m_locals)[String{ "__builtins__" }]));
	}
	return new_frame;
}

void ExecutionFrame::set_exception_to_catch(std::shared_ptr<PyObject> exception)
{
	m_exception_to_catch = std::move(exception);
}

void ExecutionFrame::set_exception(std::shared_ptr<PyObject> exception)
{
	m_exception = std::move(exception);
}

bool ExecutionFrame::catch_exception(std::shared_ptr<PyObject> exception) const
{
	if (m_exception_to_catch)
		return m_exception_to_catch.get() == exception.get();
	else
		return false;
}

void ExecutionFrame::put_local(const std::string &name, std::shared_ptr<PyObject> obj)
{
	m_locals->insert(String{ name }, obj);
}

void ExecutionFrame::put_global(const std::string &name, std::shared_ptr<PyObject> obj)
{
	m_globals->insert(String{ name }, obj);
}


const std::unique_ptr<PyDict> &ExecutionFrame::locals() const { return m_locals; }

const std::shared_ptr<PyDict> &ExecutionFrame::globals() const { return m_globals; }

const std::shared_ptr<PyModule> &ExecutionFrame::builtins() const { return m_builtins; }

std::shared_ptr<ExecutionFrame> ExecutionFrame::exit()
{
	if (m_ns) {
		for (const auto &[k, v] : m_locals->map()) {
			m_ns->insert(k, v);
		}
	}
	return m_parent;
}
