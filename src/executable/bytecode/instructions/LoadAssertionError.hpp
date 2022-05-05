#include "Instructions.hpp"


class LoadAssertionError final : public Instruction
{
	Register m_assertion_location;

  public:
	LoadAssertionError(Register assertion_location) : m_assertion_location(assertion_location) {}

	std::string to_string() const final
	{
		return fmt::format("LOAD_ASSERTION_ERROR r{}", m_assertion_location);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};
