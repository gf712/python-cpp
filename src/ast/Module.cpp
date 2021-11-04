#include "AST.hpp"

#include "bytecode/instructions/ReturnValue.hpp"

namespace ast {
Register
	Module::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	Register last{ 0 };
	for (const auto &statement : m_body) {
		last = statement->generate(function_id, generator, ctx);
	}
	generator.emit<ReturnValue>(function_id, last);
	return 0;
}
}// namespace ast
