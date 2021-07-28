#include "ast/AST.hpp"
#include "BytecodeGenerator.hpp"
#include "instructions/Instructions.hpp"


FunctionInfo::FunctionInfo(size_t function_id_, BytecodeGenerator *generator_)
	: function_id(function_id_), generator(generator_)
{
	generator->enter_function();
}

FunctionInfo::~FunctionInfo() { generator->exit_function(); }

BytecodeGenerator::BytecodeGenerator() : m_frame_register_count({ start_register })
{
	// allocate main
	allocate_function();
}

BytecodeGenerator::~BytecodeGenerator() {}

FunctionInfo BytecodeGenerator::allocate_function()
{
	m_functions.emplace_back();
	return FunctionInfo{ m_functions.size() - 1, this };
}

void BytecodeGenerator::relocate_labels(const std::vector<std::unique_ptr<Instruction>> &executable,
	const std::vector<size_t> &offsets)
{
	for (const auto &ins : executable) { ins->relocate(*this, offsets); }
}

std::shared_ptr<Bytecode> BytecodeGenerator::generate_executable()
{
	std::vector<std::unique_ptr<Instruction>> executable;
	std::vector<FunctionMetaData> functions;
	std::vector<size_t> function_offsets;
	function_offsets.resize(m_functions.size());
	size_t offset = 0;

	for (size_t i = 1; i < m_functions.size(); ++i) {
		for (auto &&ins : m_functions[i]) { executable.push_back(std::move(ins)); }
		functions.push_back(FunctionMetaData{
			offset, std::to_string(i), m_function_register_count[m_functions.size() - i] });
		function_offsets[i] = offset;
		offset += m_functions[i].size();
	}

	// make sure that at the end of compiling code we are back to __main__ frame
	ASSERT(m_frame_register_count.size() == 1)

	for (auto &&ins : m_functions[0]) { executable.push_back(std::move(ins)); }
	functions.push_back(FunctionMetaData{ offset, "__main__", m_frame_register_count.back() });
	function_offsets[0] = offset;

	relocate_labels(executable, function_offsets);

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
	std::vector<FunctionMetaData> &&funcs,
	size_t main_local_register_count)
	: m_instructions(std::move(ins)), m_functions(std::move(funcs)),
	  m_main_local_register_count(main_local_register_count)
{}


std::string Bytecode::to_string() const
{
	size_t address{ 0 };
	std::ostringstream os;
	size_t func_idx = 0;
	for (const auto &function_metadata : m_functions) {
		size_t function_end;
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