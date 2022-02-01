#include "StoreName.hpp"

#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"


void StoreName::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &value = vm.reg(m_source);
	interpreter.store_object(m_object_name, value);
}