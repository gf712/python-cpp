#include "JumpIfTrue.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyNone.hpp"

using namespace py;

PyResult JumpIfTrue::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto &result = vm.reg(m_test_register);

	const auto test_result = truthy(result, interpreter);
	if (test_result.is_ok()) {
		if ((std::holds_alternative<PyObject *>(test_result.unwrap())
				&& test_result.unwrap_as<PyObject>() == py_true())
			|| (std::holds_alternative<NameConstant>(test_result.unwrap())
				&& std::get<bool>(std::get<NameConstant>(test_result.unwrap()).value))) {
			{
				const auto ip = vm.instruction_pointer() + m_label->position();
				vm.set_instruction_pointer(ip);
			}
		}
	}
	return test_result;
}

void JumpIfTrue::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_label->set_position(m_label->position() - instruction_idx - 1);
	m_label->immutable();
}

std::vector<uint8_t> JumpIfTrue::serialize() const
{
	ASSERT(m_label->position() < std::numeric_limits<uint8_t>::max())
	if (m_offset.has_value()) { ASSERT(m_offset < std::numeric_limits<uint8_t>::max()) }

	return {
		JUMP_IF_TRUE,
		m_test_register,
		static_cast<uint8_t>(m_label->position()),
		m_offset ? uint8_t{ 0 } : static_cast<uint8_t>(*m_offset),
	};
}