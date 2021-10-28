#include "ast/AST.hpp"
#include "ast/optimizers/ConstantFolding.hpp"

#include "BytecodeGenerator.hpp"
#include "instructions/Instructions.hpp"


FunctionInfo::FunctionInfo(size_t function_id_, BytecodeGenerator *generator_)
	: function_id(function_id_), generator(generator_)
{
	generator->enter_function();
}

FunctionInfo::~FunctionInfo() { generator->exit_function(function_id); }

BytecodeGenerator::BytecodeGenerator() : m_frame_register_count({ start_register })
{
	// allocate main
	allocate_function();
}

BytecodeGenerator::~BytecodeGenerator() {}

FunctionInfo BytecodeGenerator::allocate_function()
{
	auto &new_func = m_functions.emplace_back();
	new_func.metadata.function_name = std::to_string(m_functions.size() - 1);
	return FunctionInfo{ m_functions.size() - 1, this };
}

void BytecodeGenerator::relocate_labels(const FunctionBlocks &functions,
	const std::vector<size_t> &offsets)
{
	for (const auto &block : functions) {
		for (const auto &ins : block.instructions) { ins->relocate(*this, offsets); }
	}
}

std::shared_ptr<Bytecode> BytecodeGenerator::generate_executable()
{
	// make sure that at the end of compiling code we are back to __main__ frame
	ASSERT(m_frame_register_count.size() == 1)

	FunctionBlocks functions;
	std::vector<size_t> function_offsets;
	function_offsets.resize(m_functions.size());
	size_t offset = 0;
	spdlog::debug("m_functions size: {}", m_functions.size());
	for (size_t i = 1; i < m_functions.size(); ++i) {
		auto &new_function_block = functions.emplace_back(std::move(m_functions[i]));
		spdlog::debug("function {} requires {} virtual registers",
			new_function_block.metadata.function_name,
			new_function_block.metadata.register_count);
		new_function_block.metadata.offset = offset;
		function_offsets[i] = offset;
		offset += new_function_block.instructions.size();
	}

	auto &main_function_block = functions.emplace_back(std::move(m_functions.front()));
	main_function_block.metadata.offset = offset;
	function_offsets.front() = offset;
	spdlog::debug(
		"__main__ requires {} virtual registers", main_function_block.metadata.register_count);

	relocate_labels(functions, function_offsets);

	return std::make_shared<Bytecode>(std::move(functions));
}


std::shared_ptr<Bytecode> BytecodeGenerator::compile(std::shared_ptr<ast::ASTNode> node, compiler::OptimizationLevel lvl)
{
	if (lvl > compiler::OptimizationLevel::None) {
		ast::optimizer::constant_folding(node);
	}
	auto generator = BytecodeGenerator();
	ast::ASTContext ctx;
	node->generate_impl(0, generator, ctx);
	// allocate registers for __main__
	generator.m_functions.front().metadata.register_count = generator.register_count();
	return generator.generate_executable();
}


Bytecode::Bytecode(FunctionBlocks &&func_blocks)
{
	for (size_t i = 0; i < func_blocks.size() - 1; ++i) {
		auto &func = func_blocks.at(i);
		for (auto &&ins : func.instructions) { m_instructions.push_back(std::move(ins)); }
		m_functions.push_back(std::move(func.metadata));
	}
	auto &main_func = func_blocks.back();
	for (auto &&ins : main_func.instructions) { m_instructions.push_back(std::move(ins)); }
	m_functions.push_back(std::move(main_func.metadata));

	m_main_local_register_count = main_func.metadata.register_count;
}


std::string Bytecode::to_string() const
{
	size_t address{ 0 };
	std::ostringstream os;
	size_t func_idx = 0;
	for (const auto &function_metadata : m_functions) {
		size_t function_end{ 0 };
		if (func_idx < m_functions.size() - 1) {
			function_end = m_functions[func_idx + 1].offset;
		} else {
			function_end = m_instructions.size();
		}
		os << function_metadata.function_name << ":\n";
		for (size_t i = function_metadata.offset; i < function_end; ++i) {
			const auto &ins = m_instructions.at(i);
			os << fmt::format("{:>5x} {}", address, ins->to_string()) << '\n';
			address += sizeof(Instruction);
		}
		func_idx++;
	}

	return os.str();
}

void Bytecode::add_instructions(std::unique_ptr<Instruction> &&ins)
{
	m_instructions.push_back(std::move(ins));
}
