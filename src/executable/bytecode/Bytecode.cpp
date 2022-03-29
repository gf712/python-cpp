#include "Bytecode.hpp"
#include "instructions/Instructions.hpp"
#include "serialization/deserialize.hpp"
#include "serialization/serialize.hpp"

Bytecode::Bytecode(size_t register_count,
	size_t stack_size,
	std::string function_name,
	InstructionVector &&instructions,
	std::vector<View> block_views)
	: Function(register_count, stack_size, function_name, FunctionExecutionBackend::BYTECODE),
	  m_instructions(std::move(instructions)), m_block_views(block_views)
{}

std::string Bytecode::to_string() const
{
	std::ostringstream os;
	size_t block_idx{ 0 };
	for (const auto &block : m_block_views) {
		os << "- block " << block_idx++ << ":\n";
		for (const auto &ins : block) {
			os << fmt::format("    {} {}", (void *)ins.get(), ins->to_string()) << '\n';
		}
	}

	return os.str();
}

std::vector<uint8_t> Bytecode::serialize() const
{
	std::vector<uint8_t> result;

	py::serialize(m_register_count, result);
	py::serialize(m_stack_size, result);
	py::serialize(m_function_name, result);
	py::serialize(static_cast<uint8_t>(m_backend), result);

	const size_t block_count = m_block_views.size();
	py::serialize(block_count, result);
	for (const auto &block : m_block_views) {
		const size_t block_size = block.end() - block.begin();
		py::serialize(block_size, result);

		for (const auto &ins : block) {
			auto serialized_instruction = ins->serialize();
			result.insert(
				result.end(), serialized_instruction.begin(), serialized_instruction.end());
		}
	}
	return result;
}

std::unique_ptr<Bytecode> Bytecode::deserialize(std::span<const uint8_t> &buffer)
{
	const auto register_count = py::deserialize<size_t>(buffer);
	const auto stack_size = py::deserialize<size_t>(buffer);
	const auto function_name = py::deserialize<std::string>(buffer);
	const auto backend = static_cast<FunctionExecutionBackend>(py::deserialize<uint8_t>(buffer));
	(void)backend;

	InstructionVector instructions;
	std::vector<View> block_views;
	const auto block_count = py::deserialize<size_t>(buffer);

	for (size_t i = 0, ins_index_in_block_count = 0; i < block_count; ++i) {
		const auto block_size = py::deserialize<size_t>(buffer);
		for (size_t ins_index_in_block = 0; ins_index_in_block < block_size; ++ins_index_in_block) {
			instructions.push_back(::deserialize(buffer));
		}
		InstructionVector::const_iterator start = instructions.begin() + ins_index_in_block_count;
		InstructionVector::const_iterator end = instructions.end();
		block_views.emplace_back(start, end);
		ins_index_in_block_count = instructions.size();
	}

	return std::make_unique<Bytecode>(
		register_count, stack_size, function_name, std::move(instructions), block_views);
}
