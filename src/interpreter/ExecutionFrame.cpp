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