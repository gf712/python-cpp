#include "Bytecode.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "executable/bytecode/codegen/BytecodeGenerator.hpp"

#include "lexer/Lexer.hpp"
#include "parser/Parser.hpp"

#include "gtest/gtest.h"


// FIXME: think about what should be tested here
// namespace {
// std::shared_ptr<BytecodeProgram> generate_bytecode_executable(std::string_view program)
// {
// 	auto lexer = Lexer::create(std::string(program), "_bytecode_tests_.py");
// 	parser::Parser p{ lexer };
// 	p.parse();

// 	auto module = as<ast::Module>(p.module());
// 	ASSERT(module);

// 	auto bytecode =
// 		codegen::BytecodeGenerator::compile(module, {}, compiler::OptimizationLevel::None);
// 	return std::static_pointer_cast<BytecodeProgram>(bytecode);
// }
// }// namespace

// TEST(Bytecode, CreatesExecutableWithOnlyMain)
// {
// 	static constexpr std::string_view program = "print(\"Hello, world!\")\n";

// 	auto bytecode = generate_bytecode_executable(program);
// 	const auto entry_instruction = bytecode->begin();
// 	const auto return_instruction = bytecode->end();

// 	ASSERT_EQ(std::distance(entry_instruction, return_instruction), 4);
// }

// TEST(Bytecode, CreatesExecutableWithMultipleFunctionDefinitions)
// {
// 	constexpr std::string_view program =
// 		"def foo(arg):\n"
// 		"   return arg + 42\n"
// 		"def bar(arg):\n"
// 		"   return arg\n";

// 	auto bytecode = generate_bytecode_executable(program);
// 	{
// 		const auto entry_instruction = bytecode->begin();
// 		const auto return_instruction = bytecode->end();

// 		ASSERT_EQ(std::distance(entry_instruction, return_instruction), 3);
// 	}
// 	// {
// 	// 	const auto &foo = bytecode->as_pyfunction("foo");
// 	// 	ASSERT_EQ(foo->backend(), FunctionExecutionBackend::BYTECODE);
// 	// 	const auto entry_instruction = std::static_pointer_cast<Bytecode>(foo)->begin();
// 	// 	const auto return_instruction = std::static_pointer_cast<Bytecode>(foo)->end();
// 	// 	ASSERT_EQ(std::distance(entry_instruction, return_instruction), 6);
// 	// }
// 	// {
// 	// 	const auto &bar = bytecode->as_pyfunction("bar");
// 	// 	ASSERT_EQ(bar->backend(), FunctionExecutionBackend::BYTECODE);
// 	// 	const auto entry_instruction = std::static_pointer_cast<Bytecode>(bar)->begin();
// 	// 	const auto return_instruction = std::static_pointer_cast<Bytecode>(bar)->end();
// 	// 	ASSERT_EQ(std::distance(entry_instruction, return_instruction), 4);
// 	// }
// }
