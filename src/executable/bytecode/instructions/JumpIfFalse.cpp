#include "JumpIfFalse.hpp"
#include "executable/Label.hpp"
#include "runtime/PyBool.hpp"
#include "vm/VM.hpp"

#include "../serialization/serialize.hpp"

using namespace py;

PyResult<Value> JumpIfFalse::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(m_offset.has_value());
	auto &result = vm.reg(m_test_register);

	const auto test_result = [&] {
		[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
		return truthy(result, interpreter);
	}();
	if (test_result.is_ok() && test_result.unwrap() == false) {
		const auto ip = vm.instruction_pointer() + *m_offset;
		vm.set_instruction_pointer(ip);
	} else if (test_result.is_err()) {
		return Err(test_result.unwrap_err());
	}
	return Ok(Value{ NameConstant{ test_result.unwrap() } });
}


void JumpIfFalse::relocate(size_t instruction_idx)
{
	m_offset = m_label->position() - instruction_idx - 1;
}

std::vector<uint8_t> JumpIfFalse::serialize() const
{
	ASSERT(m_offset.has_value());

	std::vector<uint8_t> result{
		JUMP_IF_FALSE,
		m_test_register,
	};
	::serialize(*m_offset, result);
	return result;
}
