#include "Jump.hpp"
#include "runtime/PyNone.hpp"

using namespace py;

PyResult<Value> Jump::execute(VirtualMachine &vm, Interpreter &) const
{
	ASSERT(m_offset.has_value())
	const auto ip = vm.instruction_pointer() + *m_offset;
	vm.set_instruction_pointer(ip);
	return Ok(Value{ py_none() });
};


void Jump::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_offset = m_label->position() - instruction_idx - 1;
}

std::vector<uint8_t> Jump::serialize() const
{
	ASSERT(m_offset.has_value())
	ASSERT(m_offset < std::numeric_limits<uint8_t>::max())

	return {
		JUMP,
		static_cast<uint8_t>(*m_offset),
	};
}