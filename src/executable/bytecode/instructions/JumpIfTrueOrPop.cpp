#include "JumpIfTrueOrPop.hpp"
#include "runtime/PyBool.hpp"

using namespace py;

PyResult<Value> JumpIfTrueOrPop::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &result = vm.reg(m_test_register);

	const auto test_result =
		std::visit(overloaded{ [](PyObject *const &obj) -> PyResult<bool> {
								  ASSERT(obj)
								  return obj->bool_();
							  },
					   [](const auto &) -> PyResult<bool> {
						   TODO();
						   return Ok(false);
					   },
					   [](const NameConstant &value) -> PyResult<bool> {
						   if (auto *bool_type = std::get_if<bool>(&value.value)) {
							   return Ok(*bool_type);
						   } else {
							   return Ok(false);
						   }
					   } },
			result);

	if (test_result.is_err()) { return Err(test_result.unwrap_err()); }
	if (test_result.unwrap()) {
		const auto ip = vm.instruction_pointer() + m_label->position();
		vm.set_instruction_pointer(ip);
	}
	vm.reg(m_result_register) = NameConstant{ test_result.unwrap() };
	return Ok(Value{ NameConstant{ test_result.unwrap() } });
}

void JumpIfTrueOrPop::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_label->set_position(m_label->position() - instruction_idx - 1);
	m_label->immutable();
}

std::vector<uint8_t> JumpIfTrueOrPop::serialize() const
{
	ASSERT(m_label->position() < std::numeric_limits<uint8_t>::max())
	if (m_offset.has_value()) { ASSERT(m_offset < std::numeric_limits<uint8_t>::max()) }

	return {
		JUMP_IF_TRUE_OR_POP,
		m_test_register,
		m_result_register,
		static_cast<uint8_t>(m_label->position()),
		m_offset ? uint8_t{ 0 } : static_cast<uint8_t>(*m_offset),
	};
}