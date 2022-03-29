#include "../BytecodeProgram.hpp"
#include "BytecodeGenerator.hpp"
#include "lexer/Lexer.hpp"
#include "parser/Parser.hpp"

#include "gtest/gtest.h"


namespace {
std::shared_ptr<BytecodeProgram> generate_bytecode(std::string_view program)
{
	auto lexer = Lexer::create(std::string(program), "_bytecode_generator_tests_.py");
	parser::Parser p{ lexer };
	p.parse();

	auto module = as<ast::Module>(p.module());
	ASSERT(module)

	return std::unique_ptr<BytecodeProgram>(static_cast<BytecodeProgram *>(
		codegen::BytecodeGenerator::compile(module, {}, compiler::OptimizationLevel::None)
			.release()));
}
}// namespace

TEST(BytecodeGenerator, EmitsMainProgram)
{
	static constexpr std::string_view program = "print(\"Hello, world!\")\n";

	auto bytecode_generator = generate_bytecode(program);
	ASSERT_EQ(bytecode_generator->functions().size(), 0);
	ASSERT_TRUE(bytecode_generator->main_function());
}


TEST(BytecodeGenerator, EmitsProgramWithFunctionDefinitions)
{
	constexpr std::string_view program =
		"def foo(arg):\n"
		"   return arg + 42\n"
		"def bar(arg):\n"
		"   return arg\n";

	auto bytecode_generator = generate_bytecode(program);
	ASSERT_EQ(bytecode_generator->functions().size(), 2);
	ASSERT_TRUE(bytecode_generator->main_function());
}
