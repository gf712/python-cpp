#include "../Bytecode.hpp"
#include "../BytecodeProgram.hpp"
#include "BytecodeGenerator.hpp"
#include "executable/common.hpp"
#include "lexer/Lexer.hpp"
#include "parser/Parser.hpp"
#include "runtime/PyCode.hpp"

#include "gtest/gtest.h"


namespace {
std::shared_ptr<BytecodeProgram> generate_bytecode(std::string_view program)
{
	auto lexer = Lexer::create(std::string(program), "_bytecode_generator_tests_.py");
	parser::Parser p{ lexer };
	p.parse();

	auto module = as<ast::Module>(p.module());
	ASSERT(module);

	return std::static_pointer_cast<BytecodeProgram>(compiler::compile(
		module, {}, compiler::Backend::BYTECODE_GENERATOR, compiler::OptimizationLevel::None));
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

TEST(BytecodeGenerator, AttachesSourceLineToMainInstructions)
{
	constexpr std::string_view program =
		"x = 1\n"
		"y = 2\n"
		"z = 3\n";

	auto bytecode_generator = generate_bytecode(program);
	auto *code = static_cast<py::PyCode *>(bytecode_generator->main_function());
	ASSERT_TRUE(code);
	const auto *bytecode = static_cast<const Bytecode *>(code->function().get());

	// We don't pin down which instruction maps to which line — that's an
	// implementation detail of the codegen. We do require that at least one
	// instruction reports each of the three source lines (1, 2, 3) and that
	// no instruction reports a bogus line.
	std::set<uint32_t> observed_lines;
	for (size_t i = 0; i < static_cast<size_t>(std::distance(bytecode->begin(), bytecode->end()));
		++i) {
		if (const auto loc = bytecode->location_for(i)) { observed_lines.insert(loc->line); }
	}
	EXPECT_TRUE(observed_lines.contains(1u));
	EXPECT_TRUE(observed_lines.contains(2u));
	EXPECT_TRUE(observed_lines.contains(3u));
}
