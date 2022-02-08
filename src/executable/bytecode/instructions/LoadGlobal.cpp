#include "LoadGlobal.hpp"

#include "runtime/NameError.hpp"
#include "runtime/PyDict.hpp"

using namespace py;

void LoadGlobal::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &globals = interpreter.execution_frame()->globals()->map();
	if (const auto &it = globals.find(String{ m_object_name }); it != globals.end()) {
		vm.reg(m_destination) = it->second;
	} else {
		interpreter.raise_exception(name_error("name '{:s}' is not defined", m_object_name));
	}
}