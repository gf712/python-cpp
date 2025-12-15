#include "StoreDeref.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyCell.hpp"
#include "runtime/PyFrame.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> StoreDeref::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->freevars().size() > m_dst);
	interpreter.execution_frame()->freevars()[m_dst]->set_cell(vm.reg(m_src));
	return Ok(vm.reg(m_src));
}

std::vector<uint8_t> StoreDeref::serialize() const
{
	return {
		STORE_DEREF,
		m_dst,
		m_src,
	};
}
