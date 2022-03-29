#include "JumpIfFalseOrPop.hpp"
#include "runtime/PyBool.hpp"

using namespace py;

void JumpIfFalseOrPop::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &result = vm.reg(m_test_register);

	const bool test_result =
		std::visit(overloaded{ [](PyObject *const &obj) -> bool {
								  ASSERT(obj)
								  return obj->bool_() == py_true();
							  },
					   [](const auto &) -> bool {
						   TODO();
						   return false;
					   },
					   [](const NameConstant &value) -> bool {
						   if (auto *bool_type = std::get_if<bool>(&value.value)) {
							   return *bool_type;
						   } else {
							   return false;
						   }
					   } },
			result);
	vm.reg(m_result_register) = NameConstant{ test_result };
	if (!test_result) {
		const auto ip = vm.instruction_pointer() + m_label->position();
		vm.set_instruction_pointer(ip);
	}
};

void JumpIfFalseOrPop::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_label->set_position(m_label->position() - instruction_idx - 1);
	m_label->immutable();
}

std::vector<uint8_t> JumpIfFalseOrPop::serialize() const
{
	ASSERT(m_label->position() < std::numeric_limits<uint8_t>::max())
	if (m_offset.has_value()) { ASSERT(m_offset < std::numeric_limits<uint8_t>::max()) }

	return {
		JUMP_IF_FALSE_OR_POP,
		m_test_register,
		m_result_register,
		static_cast<uint8_t>(m_label->position()),
		m_offset ? uint8_t{ 0 } : static_cast<uint8_t>(*m_offset),
	};
}