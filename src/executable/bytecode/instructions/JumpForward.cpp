#include "JumpForward.hpp"

void JumpForward::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.jump_blocks(m_block_count);
}
