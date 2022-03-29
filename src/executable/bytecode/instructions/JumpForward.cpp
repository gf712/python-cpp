#include "JumpForward.hpp"

void JumpForward::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.jump_blocks(m_block_count);
}

std::vector<uint8_t> JumpForward::serialize() const
{
	ASSERT(m_block_count < std::numeric_limits<uint8_t>::max())

	return {
		JUMP_FORWARD,
		static_cast<uint8_t>(m_block_count),
	};
}