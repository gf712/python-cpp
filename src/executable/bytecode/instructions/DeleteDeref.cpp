#include "DeleteDeref.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyCell.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyNone.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> DeleteDeref::execute(VirtualMachine &, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->freevars().size() > m_src);
	interpreter.execution_frame()->freevars()[m_src]->set_cell(nullptr);
	return Ok(py_none());
}

std::vector<uint8_t> DeleteDeref::serialize() const
{
	return {
		DELETE_DEREF,
		m_src,
	};
}
