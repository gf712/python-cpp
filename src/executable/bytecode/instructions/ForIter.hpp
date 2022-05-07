#pragma once

#include "Instructions.hpp"

class ForIter final : public Instruction
{
	Register m_dst;
	Register m_src;
	std::shared_ptr<Label> m_exit_label;
	std::optional<size_t> m_offset;

  public:
	ForIter(Register dst, Register src, std::shared_ptr<Label> exit_label)
		: m_dst(dst), m_src(src), m_exit_label(std::move(exit_label))
	{}

	std::string to_string() const final
	{
		return fmt::format("FOR_ITER        r{:<3} r{:<3}", m_dst, m_src);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final;

	std::vector<uint8_t> serialize() const final;
};
