#include "Instructions.hpp"


class LoadException final : public Instruction
{
	Register m_destination;

  public:
	LoadException(Register destination) : m_destination(destination) {}

	std::string to_string() const final
	{
		return fmt::format("LOAD_EXCEPTION  r{}", m_destination);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return LOAD_EXCEPTION; }
};
