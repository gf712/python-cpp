#include "JumpIfFalseOrPop.hpp"
#include "executable/Label.hpp"
#include "runtime/PyBool.hpp"
#include "vm/VM.hpp"

#include "../serialization/serialize.hpp"

using namespace py;

PyResult<Value> JumpIfFalseOrPop::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(m_offset.has_value())
	auto &result = vm.reg(m_test_register);

	const auto test_result = truthy(result, interpreter);

	if (test_result.is_err()) { return Err(test_result.unwrap_err()); }
	if (!test_result.unwrap()) {
		const auto ip = vm.instruction_pointer() + *m_offset;
		vm.set_instruction_pointer(ip);
		vm.reg(m_result_register) = result;
		return Ok(result);
	} else {
		return Ok(nullptr);
	}
}

void JumpIfFalseOrPop::relocate(size_t instruction_idx)
{
	m_offset = m_label->position() - instruction_idx - 1;
}

std::vector<uint8_t> JumpIfFalseOrPop::serialize() const
{
	ASSERT(m_offset.has_value())

	std::vector<uint8_t> result{
		JUMP_IF_FALSE_OR_POP,
		m_test_register,
		m_result_register,
	};

	::serialize(*m_offset, result);

	return result;
}
