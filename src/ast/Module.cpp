#include "AST.hpp"

#include "bytecode/instructions/Instructions.hpp"

namespace ast {
Register Module::generate(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	Register last;
	for (const auto &statement : m_body) {
		last = statement->generate(function_id, generator, ctx);
	}
	generator.emit<ReturnValue>(function_id, last);
	return 0;
}
}// namespace ast
