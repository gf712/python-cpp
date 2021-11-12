#include "LoadName.hpp"

#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"


void LoadName::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	if (auto value = interpreter.get_object(m_object_name)) { vm.reg(m_destination) = *value; }
}