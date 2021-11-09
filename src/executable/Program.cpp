#include "Program.hpp"


Program::Program(FunctionBlocks &&func_blocks, std::string filename)
	: m_filename(std::move(filename))
{
	const auto instruction_count = std::accumulate(func_blocks.begin(),
		func_blocks.end(),
		0u,
		[](const size_t &lhs, const FunctionBlock &rhs) { return lhs + rhs.instructions.size(); });
	// have to reserve instruction vector to avoid relocations
	// since the iterators depend on the vector memory layout
	m_instructions.reserve(instruction_count);

	auto &main_func = func_blocks.front();

	for (auto &&ins : main_func.instructions) { m_instructions.push_back(std::move(ins)); }

	const auto start_instruction = m_instructions.end() - main_func.instructions.size();
	const auto end_instruction = m_instructions.end();
	m_main_function = std::make_shared<Bytecode>(main_func.metadata.register_count,
		main_func.metadata.function_name,
		start_instruction,
		end_instruction);


	for (size_t i = 1; i < func_blocks.size(); ++i) {
		auto &func = func_blocks.at(i);
		for (auto &&ins : func.instructions) { m_instructions.push_back(std::move(ins)); }
		const auto start_instruction = m_instructions.end() - func.instructions.size();
		const auto end_instruction = m_instructions.end();

		auto bytecode = std::make_shared<Bytecode>(func.metadata.register_count,
			func.metadata.function_name,
			start_instruction,
			end_instruction);

		m_functions.emplace_back(std::move(bytecode));
	}
}

std::string Program::to_string() const
{
	std::stringstream ss;
	for (const auto &func : m_functions) {
		ss << func->function_name() << ":\n";
		ss << func->to_string() << '\n';
	}

	ss << "main:\n";
	ss << m_main_function->to_string() << '\n';
	return ss.str();
}