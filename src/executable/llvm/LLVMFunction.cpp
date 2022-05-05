#include "LLVMGenerator.hpp"
#include "runtime/Value.hpp"

using namespace codegen;

py::PyResult<py::Value> LLVMFunction::call(VirtualMachine &, Interpreter &) const
{
	// function calls with LLVM functions use a different code path since they currently
	// invoked outside the VM and/or Interpreter
	TODO();
}