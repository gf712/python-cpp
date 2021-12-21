#include "Bytecode.hpp"
#include "instructions/Instructions.hpp"

Bytecode::Bytecode(size_t registers_needed,
	std::string m_function_name,
	std::vector<View> block_views)
	: Function(registers_needed, m_function_name, FunctionExecutionBackend::BYTECODE),
	  m_block_views(block_views)
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