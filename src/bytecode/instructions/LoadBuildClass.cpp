#include "LoadBuildClass.hpp"

#include "runtime/PyModule.hpp"


void LoadBuildClass::execute(VirtualMachine &vm, Interpreter &intepreter) const
{
	vm.reg(m_dst) = std::get<std::shared_ptr<PyObject>>(
		intepreter.execution_frame()->builtins()->module_definitions().at(
			PyString::create("__build_class__")));
}