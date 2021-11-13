#include "Bytecode.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/codegen/BytecodeGenerator.hpp"

#include "lexer/Lexer.hpp"
#include "parser/Parser.hpp"

#include "gtest/gtest.h"


namespace {
std::shared_ptr<Program> generate_bytecode_executable(std::string_view program)
{
	auto lexer = Lexer::create(std::string(program), "_bytecode_tests_.py");
	parser::Parser p{ lexer };
	p.parse();

	auto module = as<ast::Module>(p.module());
	ASSERT(module)

	return codegen::BytecodeGenerator::compile(module, {}, compiler::OptimizationLevel::None);
}
}// namespace


TEST(Bytecode, CreatesExecutableWithOnlyMain)
{
	static constexpr std::string_view program = "print(\"Hello, world!\")\n";

	auto bytecode = generate_bytecode_executable(program);
	const auto entry_instruction = bytecode->begin();
	const auto return_instruction = bytecode->end();

	ASSERT_EQ(std::distance(entry_instruction, return_instruction), 4);
}

TEST(Bytecode, CreatesExecutableWithMultipleFunctionDefinitions)
{
	constexpr std::string_view program =
		"def foo(arg):\n"
		"   return arg + 42\n"
		"def bar(arg):\n"
		"   return arg\n";

	auto bytecode = generate_bytecode_executable(program);
	{
		const auto entry_instruction = bytecode->begin();
		const auto return_instruction = bytecode->end();

		ASSERT_EQ(std::distance(entry_instruction, return_instruction), 3);
	}
	{
		const auto &foo = bytecode->function(1);
		ASSERT_EQ(foo->backend(), FunctionExecutionBackend::BYTECODE);
		const auto entry_instruction = std::static_pointer_cast<Bytecode>(foo)->begin();
		const auto return_instruction = std::static_pointer_cast<Bytecode>(foo)->end();
		ASSERT_EQ(std::distance(entry_instruction, return_instruction), 6);
	}
	{
		const auto &bar = bytecode->function(2);
		ASSERT_EQ(bar->backend(), FunctionExecutionBackend::BYTECODE);
		const auto entry_instruction = std::static_pointer_cast<Bytecode>(bar)->begin();
		const auto return_instruction = std::static_pointer_cast<Bytecode>(bar)->end();
		ASSERT_EQ(std::distance(entry_instruction, return_instruction), 4);
	}
}
