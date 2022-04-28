#include "Jump.hpp"
#include "runtime/PyNone.hpp"

using namespace py;

PyResult Jump::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto ip = vm.instruction_pointer() + m_label->position();
	vm.set_instruction_pointer(ip);
	return PyResult::Ok(py_none());
};


void Jump::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_label->set_position(m_label->position() - instruction_idx - 1);
	m_label->immutable();
}

std::vector<uint8_t> Jump::serialize() const
{
	ASSERT(m_label->position() < std::numeric_limits<uint8_t>::max())
	if (m_offset.has_value()) { ASSERT(m_offset < std::numeric_limits<uint8_t>::max()) }

	return {
		JUMP,
		static_cast<uint8_t>(m_label->position()),
		static_cast<uint8_t>(m_offset.value_or(0)),
	};
}