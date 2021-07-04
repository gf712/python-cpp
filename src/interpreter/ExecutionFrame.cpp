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