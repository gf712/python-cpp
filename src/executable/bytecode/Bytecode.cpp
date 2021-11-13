#include "Bytecode.hpp"
#include "instructions/Instructions.hpp"

Bytecode::Bytecode(size_t registers_needed,
	std::string m_function_name,
	InstructionVector::const_iterator begin,
	InstructionVector::const_iterator end)
	: Function(registers_needed, m_function_name, FunctionExecutionBackend::BYTECODE),
	  m_bytecode_view(begin, end)
{}

std::string Bytecode::to_string() const
{
	std::ostringstream os;
	for (const auto &ins : m_bytecode_view) {
		os << fmt::format("{} {}", (void *)ins.get(), ins->to_string()) << '\n';
	}

	return os.str();
}