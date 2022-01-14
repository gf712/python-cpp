#include "JumpIfTrue.hpp"
#include "runtime/PyBool.hpp"

using namespace py;

void JumpIfTrue::execute(VirtualMachine &vm, Interpreter &) const
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
	if (test_result) {
		const auto ip = vm.instruction_pointer() + m_label->position();
		vm.set_instruction_pointer(ip);
	}
}

void JumpIfTrue::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_label->set_position(m_label->position() - instruction_idx - 1);
	m_label->immutable();
}