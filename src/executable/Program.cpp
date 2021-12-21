#include "Program.hpp"


Program::Program(FunctionBlocks &&func_blocks, std::string filename, std::vector<std::string> argv)
	: m_filename(std::move(filename)), m_argv(std::move(argv))
{
	std::vector<size_t> functions_instruction_count;
	functions_instruction_count.reserve(func_blocks.size());
	for (const auto &f : func_blocks) {
		functions_instruction_count.push_back(std::transform_reduce(
			f.blocks.begin(), f.blocks.end(), 0u, std::plus<size_t>{}, [](const auto &ins) {
				return ins.size();
			}));
	}

	const auto instruction_count =
		std::accumulate(functions_instruction_count.begin(), functions_instruction_count.end(), 0u);
	// have to reserve instruction vector to avoid relocations
	// since the iterators depend on the vector memory layout
	m_instructions.reserve(instruction_count);

	auto &main_func = func_blocks.front();

	std::vector<View> main_blocks;
	main_blocks.reserve(main_func.blocks.size());

	for (size_t start_idx = 0; auto &block : main_func.blocks) {
		ASSERT(!block.empty())
		for (auto &ins : block) { m_instructions.push_back(std::move(ins)); }
		InstructionVector::const_iterator start = m_instructions.cbegin() + start_idx;
		InstructionVector::const_iterator end = m_instructions.end();
		main_blocks.emplace_back(start, end);
		start_idx = m_instructions.size();
	}

	m_main_function = std::make_shared<Bytecode>(
		main_func.metadata.register_count, main_func.metadata.function_name, main_blocks);

	for (size_t i = 1; i < func_blocks.size(); ++i) {
		auto &func = *std::next(func_blocks.begin(), i);
		std::vector<View> func_blocks_view;
		func_blocks_view.reserve(func.blocks.size());
		for (size_t start_idx = m_instructions.size(); auto &block : func.blocks) {
			// ASSERT(!block.empty())
			if (block.empty()) { continue; }
			for (auto &ins : block) { m_instructions.push_back(std::move(ins)); }
			InstructionVector::const_iterator start = m_instructions.cbegin() + start_idx;
			InstructionVector::const_iterator end = m_instructions.end();
			func_blocks_view.emplace_back(start, end);
			start_idx = m_instructions.size();
		}

		auto bytecode = std::make_shared<Bytecode>(
			func.metadata.register_count, func.metadata.function_name, func_blocks_view);

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

std::string FunctionBlock::to_string() const
{
	std::ostringstream os;
	os << "Function name: " << metadata.function_name << '\n';
	size_t block_idx{ 0 };
	for (const auto &block : blocks) {
		os << "  block " << block_idx++ << '\n';
		for (const auto &ins : block) { os << "    " << ins->to_string() << '\n'; }
	}
	return os.str();
}