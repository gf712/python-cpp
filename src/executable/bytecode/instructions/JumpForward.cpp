#include "JumpForward.hpp"
#include "runtime/PyNone.hpp"

using namespace py;

PyResult JumpForward::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.jump_blocks(m_block_count);
	return PyResult::Ok(py_none());
}

std::vector<uint8_t> JumpForward::serialize() const
{
	ASSERT(m_block_count < std::numeric_limits<uint8_t>::max())

	return {
		JUMP_FORWARD,
		static_cast<uint8_t>(m_block_count),
	};
}