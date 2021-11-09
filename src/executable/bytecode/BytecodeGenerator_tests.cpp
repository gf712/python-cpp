#include "BytecodeGenerator.hpp"

#include "lexer/Lexer.hpp"
#include "parser/Parser.hpp"

#include "gtest/gtest.h"


namespace {
std::unique_ptr<BytecodeGenerator> generate_bytecode(std::string_view program)
{
	auto lexer = Lexer::create(std::string(program), "_bytecode_generator_tests_.py");
	parser::Parser p{ lexer };
	p.parse();

	auto module = as<ast::Module>(p.module());
	ASSERT(module)

	auto generator = std::make_unique<BytecodeGenerator>();

	ast::ASTContext ctx;
	module->generate_impl(0, *generator, ctx);

	return generator;
}
}

TEST(BytecodeGenerator, EmitsMainProgram)
{
	static constexpr std::string_view program = "print(\"Hello, world!\")\n";

	auto bytecode_generator = generate_bytecode(program);
	ASSERT_EQ(bytecode_generator->functions().size(), 1);

	const auto &main = bytecode_generator->function(0);
	ASSERT_EQ(main.size(), 4);
}


TEST(BytecodeGenerator, EmitsProgramWithFunctionDefinition)
{
	constexpr std::string_view program =
		"def foo(arg):\n"
		"   return arg + 42\n"
        "def bar(arg):\n"
		"   return arg\n";

	auto bytecode_generator = generate_bytecode(program);
	ASSERT_EQ(bytecode_generator->functions().size(), 3);
}


TEST(BytecodeGenerator, GeneratesCorrectLabels)
{
	constexpr std::string_view program =
		"if foo == 42:\n"
		"  print(\"It's foo!\")\n"
		"else:\n"
		"  print(\"Not foo..\")\n";

	auto bytecode_generator = generate_bytecode(program);
	ASSERT_EQ(bytecode_generator->functions().size(), 1);

	ASSERT_EQ(bytecode_generator->labels().size(), 2);
}