export module py.runtime:bytecode;
import :value;
import :executable_function;
import :executable_program;
import :executable_functionblock;
import std;

export class VirtualMachine;


export class Bytecode : public Function
{
	const InstructionVector m_instructions;
	const std::vector<InstructionSourceLocation> m_instruction_locations;

  public:
	Bytecode(std::size_t register_count,
		std::size_t locals_count,
		std::size_t stack_size,
		std::string function_name,
		InstructionVector instructions,
		std::vector<InstructionSourceLocation> instruction_locations,
		std::shared_ptr<Program> program);

	~Bytecode() override;

	auto begin() const { return m_instructions.begin(); }
	auto end() const { return m_instructions.end(); }

	std::optional<InstructionSourceLocation> location_for(std::size_t instruction_index) const;

	std::string to_string() const override;

	std::vector<std::uint8_t> serialize() const override;

	static std::unique_ptr<Bytecode> deserialize(std::span<const std::uint8_t> &buffer,
		std::shared_ptr<Program> program);

	py::PyResult<py::Value> call(VirtualMachine &, Interpreter &) const override;
	py::PyResult<py::Value> call_without_setup(VirtualMachine &, Interpreter &) const override;

	py::PyResult<py::Value> eval_loop(VirtualMachine &, Interpreter &) const;
};
