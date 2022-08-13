#include "Jump.hpp"
#include "executable/Label.hpp"
#include "runtime/PyNone.hpp"
#include "vm/VM.hpp"

#include "../serialization/serialize.hpp"

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

	std::vector<uint8_t> result{
		JUMP,
	};

	::serialize(*m_offset, result);

	return result;
}