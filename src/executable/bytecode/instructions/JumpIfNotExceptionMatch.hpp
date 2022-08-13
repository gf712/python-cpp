#include "Instructions.hpp"


class JumpIfNotExceptionMatch final : public Instruction
{
	Register m_exception_type_reg;
	std::shared_ptr<Label> m_label;
	std::optional<uint32_t> m_offset;

  public:
	JumpIfNotExceptionMatch(Register exception_type_reg, std::shared_ptr<Label> label)
		: m_exception_type_reg(exception_type_reg), m_label(std::move(label))
	{}
	JumpIfNotExceptionMatch(Register exception_type_reg, uint32_t offset)
		: m_exception_type_reg(exception_type_reg), m_offset(offset)
	{}

	std::string to_string() const final
	{
		const std::string position =
			m_offset.has_value() ? std::to_string(*m_offset) : "offset not evaluated";

		return fmt::format("JUMP_IF_NOT_EXC r{} position: {}", m_exception_type_reg, position);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final;

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return JUMP_IF_NOT_EXCEPTION_MATCH; }
};
