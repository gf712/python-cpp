#include "Jump.hpp"


void Jump::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto ip = vm.instruction_pointer() + m_label->position();
	vm.set_instruction_pointer(ip);
};


void Jump::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_label->set_position(m_label->position() - instruction_idx - 1);
	m_label->immutable();
}
