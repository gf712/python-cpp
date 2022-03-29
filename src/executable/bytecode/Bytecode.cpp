#include "Bytecode.hpp"
#include "instructions/Instructions.hpp"

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
	for (size_t i = 0; i < sizeof(size_t); ++i) {
		result.push_back(reinterpret_cast<const uint8_t *>(&m_register_count)[i]);
	}
	for (size_t i = 0; i < sizeof(size_t); ++i) {
		result.push_back(reinterpret_cast<const uint8_t *>(&m_stack_size)[i]);
	}

	const size_t function_name_size = m_function_name.size();
	for (size_t i = 0; i < sizeof(size_t); ++i) {
		result.push_back(reinterpret_cast<const uint8_t *>(&function_name_size)[i]);
	}
	for (const auto &c : m_function_name) {
		result.push_back(*reinterpret_cast<const uint8_t *>(&c));
	}
	result.push_back(static_cast<uint8_t>(m_backend));

	const size_t block_count = m_block_views.size();
	for (size_t i = 0; i < sizeof(size_t); ++i) {
		result.push_back(reinterpret_cast<const uint8_t *>(&block_count)[i]);
	}
	for (const auto &block : m_block_views) {
		const size_t block_size = block.end() - block.begin();
		for (size_t i = 0; i < sizeof(size_t); ++i) {
			result.push_back(reinterpret_cast<const uint8_t *>(&block_size)[i]);
		}
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
	size_t register_count{ 0 };
	for (size_t i = 0; i < sizeof(size_t); ++i) {
		reinterpret_cast<uint8_t *>(&register_count)[i] = buffer[i];
	}
	buffer = buffer.subspan(sizeof(size_t), buffer.size() - sizeof(size_t));

	size_t stack_size{ 0 };
	for (size_t i = 0; i < sizeof(size_t); ++i) {
		reinterpret_cast<uint8_t *>(&stack_size)[i] = buffer[i];
	}
	buffer = buffer.subspan(sizeof(size_t), buffer.size() - sizeof(size_t));

	size_t function_name_size{ 0 };
	for (size_t i = 0; i < sizeof(size_t); ++i) {
		reinterpret_cast<uint8_t *>(&function_name_size)[i] = buffer[i];
	}
	buffer = buffer.subspan(sizeof(size_t), buffer.size() - sizeof(size_t));

	std::string function_name;
	function_name.resize(function_name_size);
	for (size_t i = 0; i < function_name_size; ++i) {
		function_name[i] = *reinterpret_cast<const char *>(&buffer[i]);
	}
	buffer = buffer.subspan(function_name_size, buffer.size() - function_name_size);

	const uint8_t backend = buffer.front();
	(void)backend;
	buffer = buffer.subspan(1);

	InstructionVector instructions;
	std::vector<View> block_views;
	size_t block_count{ 0 };
	for (size_t i = 0; i < sizeof(size_t); ++i) {
		reinterpret_cast<uint8_t *>(&block_count)[i] = buffer[i];
	}
	buffer = buffer.subspan(sizeof(size_t), buffer.size() - sizeof(size_t));


	for (size_t i = 0, ins_index_in_block_count = 0; i < block_count; ++i) {
		size_t block_size{ 0 };
		for (size_t j = 0; j < sizeof(size_t); ++j) {
			reinterpret_cast<uint8_t *>(&block_size)[j] = buffer[j];
		}
		buffer = buffer.subspan(sizeof(size_t), buffer.size() - sizeof(size_t));
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
