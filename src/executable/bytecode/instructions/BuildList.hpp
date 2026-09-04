#pragma once

class BuildList final : public Instruction
{
	Register m_dst;
	size_t m_size;

  public:
	BuildList(Register dst, size_t size) : m_dst(dst), m_size(size) {}

	std::string to_string() const final
	{
		return std::format("BUILD_LIST      r{:<3} (size={})", m_dst, m_size);
	}

	py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return BUILD_LIST; }
};
