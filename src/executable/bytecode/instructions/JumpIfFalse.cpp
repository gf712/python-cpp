#include "JumpIfFalse.hpp"
#include "runtime/PyBool.hpp"

using namespace py;

void JumpIfFalse::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &result = vm.reg(m_test_register);

	const bool test_result = std::visit(
		overloaded{ [](PyObject *const &obj) -> bool { return obj->bool_() == py_true(); },
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
	if (!test_result) {
		const auto ip = vm.instruction_pointer() + m_label->position();
		vm.set_instruction_pointer(ip);
	}
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