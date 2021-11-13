
#include "Bytecode.hpp"

#include "BytecodeGenerator.hpp"
#include "ast/AST.hpp"
#include "ast/optimizers/ConstantFolding.hpp"
#include "executable/FunctionBlock.hpp"
#include "executable/Program.hpp"
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

void BytecodeGenerator::exit_function(size_t function_id)
{
	m_functions[function_id].metadata.register_count = register_count();
	m_frame_register_count.pop_back();
}

FunctionInfo BytecodeGenerator::allocate_function()
{
	auto &new_func = m_functions.emplace_back();
	new_func.metadata.function_name = std::to_string(m_functions.size() - 1);
	return FunctionInfo{ m_functions.size() - 1, this };
}

void BytecodeGenerator::relocate_labels(const FunctionBlocks &functions)
{
	for (const auto &block : functions) {
		(void)block;
		// size_t instruction_idx{ 0 };
		// for (const auto &ins : block.instructions) { ins->relocate(*this, instruction_idx++); }
	}
}

std::shared_ptr<Program> BytecodeGenerator::generate_executable(std::string filename,
	std::vector<std::string> argv)
{
	ASSERT(m_frame_register_count.size() == 1)
	// make sure that at the end of compiling code we are back to __main__ frame
	// ASSERT(m_frame_register_count.size() == 1)

	// std::vector<std::unique_ptr<Instruction>> instructions;
	// for (const auto &function : m_functions) {
	// 	for (auto &&ins : function.instructions) { instructions.push_back(std::move(ins)); }
	// }


	// FunctionBlocks functions;
	// std::vector<size_t> function_offsets;
	// function_offsets.resize(m_functions.size());
	// size_t offset = 0;
	// spdlog::debug("m_functions size: {}", m_functions.size());
	// for (size_t i = 1; i < m_functions.size(); ++i) {
	// 	auto &new_function_block = functions.emplace_back(std::move(m_functions[i]));
	// 	spdlog::debug("function {} requires {} virtual registers",
	// 		new_function_block.metadata.function_name,
	// 		new_function_block.metadata.register_count);
	// 	new_function_block.metadata.offset = offset;
	// 	function_offsets[i] = offset;
	// 	offset += new_function_block.instructions.size();
	// }

	// auto &main_function_block = functions.emplace_back(std::move(m_functions.front()));
	// main_function_block.metadata.offset = offset;
	// function_offsets.front() = offset;
	// spdlog::debug(
	// 	"__main__ requires {} virtual registers", main_function_block.metadata.register_count);

	relocate_labels(m_functions);

	return std::make_shared<Program>(std::move(m_functions), filename, argv);
}


std::shared_ptr<Program> BytecodeGenerator::compile(std::shared_ptr<ast::ASTNode> node,
	std::vector<std::string> argv,
	compiler::OptimizationLevel lvl)
{
	auto module = as<ast::Module>(node);
	ASSERT(module)

	if (lvl > compiler::OptimizationLevel::None) { ast::optimizer::constant_folding(node); }

	auto generator = BytecodeGenerator();

	ast::ASTContext ctx;
	node->generate_impl(0, generator, ctx);
	// allocate registers for __main__
	generator.m_functions.front().metadata.register_count = generator.register_count();
	auto executable = generator.generate_executable(module->filename(), argv);
	return executable;
}
