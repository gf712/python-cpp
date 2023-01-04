#include "Instructions.hpp"

class LeaveExceptionHandling final : public Instruction
{
  public:
	LeaveExceptionHandling() = default;

	std::string to_string() const final { return fmt::format("LEAVE_EXC_HANDLE"); }

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return LEAVE_EXCEPTION_HANDLING; }
};
