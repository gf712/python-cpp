#include "JumpIfFalse.hpp"
#include "runtime/PyBool.hpp"

using namespace py;

PyResult<Value> JumpIfFalse::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &result = vm.reg(m_test_register);

	const auto test_result =
		std::visit(overloaded{ [](PyObject *const &obj) -> PyResult<bool> { return obj->bool_(); },
					   [](const auto &) -> PyResult<bool> {
						   TODO();
						   return Err(nullptr);
					   },
					   [](const NameConstant &value) -> PyResult<bool> {
						   if (auto *bool_type = std::get_if<bool>(&value.value)) {
							   return Ok(*bool_type);
						   } else {
							   return Ok(false);
						   }
					   } },
			result);
	if (test_result.is_ok() && test_result.unwrap() == false) {
		const auto ip = vm.instruction_pointer() + m_label->position();
		vm.set_instruction_pointer(ip);
	}
	if (test_result.is_err()) return Err(test_result.unwrap_err());
	return Ok(Value{ NameConstant{ test_result.unwrap() } });
};


void JumpIfFalse::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_label->set_position(m_label->position() - instruction_idx - 1);
	m_label->immutable();
}

std::vector<uint8_t> JumpIfFalse::serialize() const
{
	ASSERT(m_label->position() < std::numeric_limits<uint8_t>::max())
	if (m_offset.has_value()) { ASSERT(m_offset < std::numeric_limits<uint8_t>::max()) }

	return {
		JUMP_IF_FALSE,
		static_cast<uint8_t>(m_label->position()),
		static_cast<uint8_t>(m_offset.value_or(0)),
	};
}