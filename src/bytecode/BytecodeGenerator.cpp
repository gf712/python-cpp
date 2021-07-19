#include "ast/AST.hpp"
#include "BytecodeGenerator.hpp"
#include "instructions/Instructions.hpp"

BytecodeGenerator::BytecodeGenerator()
{
	// allocate main
	allocate_function();
}

BytecodeGenerator::~BytecodeGenerator() {}

size_t BytecodeGenerator::allocate_function()
{
	m_functions.emplace_back();
	return m_functions.size() - 1;
}

void BytecodeGenerator::rellocate_labels(
	const std::vector<std::unique_ptr<Instruction>> &executable,
	const std::vector<size_t> &offsets)
{
	for (const auto &ins : executable) { ins->rellocate(*this, offsets); }
}

std::shared_ptr<Bytecode> BytecodeGenerator::generate_executable()
{
	std::vector<std::unique_ptr<Instruction>> executable;
	std::vector<std::pair<size_t, std::string>> functions;
	std::vector<size_t> function_offsets;
	function_offsets.resize(m_functions.size());
	size_t offset = 0;

	for (size_t i = 1; i < m_functions.size(); ++i) {
		for (auto &&ins : m_functions[i]) { executable.push_back(std::move(ins)); }
		functions.emplace_back(offset, std::to_string(i));
		function_offsets[i] = offset;
		offset += m_functions[i].size();
	}
	for (auto &&ins : m_functions[0]) { executable.push_back(std::move(ins)); }
	functions.emplace_back(offset, "__main__");
	function_offsets[0] = offset;

	rellocate_labels(executable, function_offsets);

	return std::make_shared<Bytecode>(
		std::move(executable), std::move(functions), register_count());
}


std::shared_ptr<Bytecode> BytecodeGenerator::compile(std::shared_ptr<ast::ASTNode> node)
{
	auto generator = BytecodeGenerator();
	ast::ASTContext ctx;
	node->generate(0, generator, ctx);
	return generator.generate_executable();
}


Bytecode::Bytecode(std::vector<std::unique_ptr<Instruction>> &&ins,
	std::vector<std::pair<size_t, std::string>> &&funcs,
	size_t virtual_register_count)
	: m_instructions(std::move(ins)), m_functions(std::move(funcs)),
	  m_virtual_register_count(virtual_register_count)
{}


std::string Bytecode::to_string() const
{
	size_t address{ 0 };
	std::ostringstream os;
	size_t func_idx = 0;
	for (const auto &[offset, name] : m_functions) {
		size_t function_end;
		if (func_idx < m_functions.size() - 1) {
			function_end = m_functions[func_idx + 1].first;
		} else {
			function_end = m_instructions.size();
		}
		os << name << ":\n";
		for (size_t i = offset; i < function_end; ++i) {
			const auto &ins = m_instructions.at(i);
			os << fmt::format("{:>5x} {}", address, ins->to_string()) << '\n';
			address += sizeof(Instruction);
		}
		func_idx++;
	}

	return os.str();
}