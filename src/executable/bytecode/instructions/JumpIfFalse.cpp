#include "JumpIfFalse.hpp"
#include "runtime/PyBool.hpp"

using namespace py;

PyResult JumpIfFalse::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &result = vm.reg(m_test_register);

	const auto test_result =
		std::visit(overloaded{ [](PyObject *const &obj) -> PyResult { return obj->bool_(); },
					   [](const auto &) -> PyResult {
						   TODO();
						   return PyResult::Err(nullptr);
					   },
					   [](const NameConstant &value) -> PyResult {
						   if (auto *bool_type = std::get_if<bool>(&value.value)) {
							   return PyResult::Ok(*bool_type ? py_true() : py_false());
						   } else {
							   return PyResult::Ok(py_false());
						   }
					   } },
			result);
	if (test_result.is_ok() && test_result.unwrap_as<PyObject>() == py_false()) {
		const auto ip = vm.instruction_pointer() + m_label->position();
		vm.set_instruction_pointer(ip);
	}
	return test_result;
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