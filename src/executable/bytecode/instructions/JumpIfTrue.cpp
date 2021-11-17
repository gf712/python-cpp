#include "JumpIfTrue.hpp"


void JumpIfTrue::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &result = vm.reg(m_test_register);

	const bool test_result =
		std::visit(overloaded{ [](PyObject *const &obj) -> bool {
								  if (auto bool_obj = as<PyNameConstant>(obj)) {
									  const auto value = bool_obj->value();
									  if (auto *bool_type = std::get_if<bool>(&value.value)) {
										  return *bool_type;
									  } else {
										  return false;
									  }
								  }
								  TODO()
								  return false;
							  },
					   [](const auto &) -> bool {
						   TODO()
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
	if (test_result) {
		const auto ip = vm.instruction_pointer() + m_label.position();
		vm.set_instruction_pointer(ip);
	}
}

void JumpIfTrue::relocate(codegen::BytecodeGenerator &generator, size_t instruction_idx)
{
	m_label = generator.label(m_label);
	m_label.set_position(m_label.position() - instruction_idx - 1);
}