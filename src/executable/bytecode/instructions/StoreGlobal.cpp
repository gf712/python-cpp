#include "StoreGlobal.hpp"

#include "runtime/PyObject.hpp"

using namespace py;

void StoreGlobal::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &value = vm.reg(m_source);
	interpreter.execution_frame()->put_global(m_object_name, value);
}