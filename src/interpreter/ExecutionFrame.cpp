#include "ExecutionFrame.hpp"
#include "runtime/PyObject.hpp"

std::string SymbolTable::to_string() const
{
	std::ostringstream os;
	for (const auto &[name, obj] : symbols) {
		os << name << ": ";
		if (obj) {
			os << obj->to_string();
		} else {
			os << "(Empty)";
		}
		os << '\n';
	}
	return os.str();
}

std::shared_ptr<ExecutionFrame> ExecutionFrame::create(std::shared_ptr<ExecutionFrame> parent,
	const std::string &scope_name)
{
	auto new_frame = std::shared_ptr<ExecutionFrame>(new ExecutionFrame{});
	new_frame->m_parent = std::move(parent);
	new_frame->m_symbol_table->symbols["__name__"] = PyString::from(String{ scope_name });
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