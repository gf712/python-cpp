#pragma once

#include "Instructions.hpp"

#include <optional>

class ForIter final : public Instruction
{
	Register m_dst;
	Register m_src;
	std::shared_ptr<Label> m_body_label;
	std::shared_ptr<Label> m_exit_label;
	std::optional<int32_t> m_offset;
	std::optional<int32_t> m_body_offset;

  public:
	ForIter(Register dst, Register src, std::shared_ptr<Label> exit_label)
		: m_dst(dst), m_src(src), m_exit_label(std::move(exit_label))
	{}

	ForIter(Register dst,
		Register src,
		std::shared_ptr<Label> body_label,
		std::shared_ptr<Label> exit_label)
		: m_dst(dst), m_src(src), m_body_label(std::move(body_label)),
		  m_exit_label(std::move(exit_label))
	{}

	ForIter(Register dst, Register src, int32_t offset) : ForIter(dst, src, offset, int32_t{ 0 }) {}

	ForIter(Register dst, Register src, int32_t offset, int32_t body_offset)
		: m_dst(dst), m_src(src), m_offset(offset), m_body_offset(body_offset)
	{}

	std::string to_string() const final
	{
		return fmt::format("FOR_ITER        r{:<3} r{:<3} body={}, orelse={}",
			m_dst,
			m_src,
			m_body_offset.value_or(-1),
			m_offset.value_or(-1));
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final;

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return FOR_ITER; }
};
