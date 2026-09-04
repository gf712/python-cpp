module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.
#include "../serialization/serialize.hpp"
#include "executable/Label.hpp"


using namespace py;

PyResult<Value> JumpForward::execute(VirtualMachine &vm, Interpreter &) const
{
	ASSERT(m_offset.has_value());
	const auto ip = vm.instruction_pointer() + *m_offset;
	vm.set_instruction_pointer(ip);
	return Ok(Value{ py_none() });
}


void JumpForward::relocate(size_t instruction_idx)
{
	m_offset = m_label->position() - instruction_idx - 1;
}

std::vector<uint8_t> JumpForward::serialize() const
{
	ASSERT(m_offset.has_value());

	std::vector<uint8_t> result{
		JUMP_FORWARD,
	};

	::serialize(*m_offset, result);

	return result;
}
