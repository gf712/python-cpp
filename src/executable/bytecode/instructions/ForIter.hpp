#pragma once

#include "Instructions.hpp"

#include <optional>

class ForIter final : public Instruction
{
	Register m_dst;
	Register m_src;
	std::shared_ptr<Label> m_exit_label;
	std::optional<int32_t> m_offset;

  public:
	ForIter(Register dst, Register src, std::shared_ptr<Label> exit_label)
		: m_dst(dst), m_src(src), m_exit_label(std::move(exit_label))
	{}

	ForIter(Register dst, Register src, int32_t offset) : m_dst(dst), m_src(src), m_offset(offset)
	{}

	std::string to_string() const final
	{
		return fmt::format("FOR_ITER        r{:<3} r{:<3}", m_dst, m_src);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final;

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return FOR_ITER; }
};
