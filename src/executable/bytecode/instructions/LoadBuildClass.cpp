#include "LoadBuildClass.hpp"

#include "runtime/PyModule.hpp"


void LoadBuildClass::execute(VirtualMachine &vm, Interpreter &intepreter) const
{
	vm.reg(m_dst) =
		std::get<PyObject *>(intepreter.execution_frame()->builtins()->symbol_table().at(
			PyString::create("__build_class__")));
}