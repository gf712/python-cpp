#include "Bytecode.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "executable/bytecode/codegen/BytecodeGenerator.hpp"

#include "lexer/Lexer.hpp"
#include "parser/Parser.hpp"

#include "gtest/gtest.h"

namespace {
Bytecode make_bytecode_with_locations(std::vector<InstructionSourceLocation> locations)
{
	return Bytecode{ /*register_count=*/0,
		/*locals_count=*/0,
		/*stack_size=*/0,
		/*function_name=*/"<test>",
		InstructionVector{},
		std::move(locations),
		/*program=*/nullptr };
}
}// namespace

TEST(BytecodeLocationFor, ReturnsNulloptWhenTableIsEmpty)
{
	auto bc = make_bytecode_with_locations({});
	EXPECT_FALSE(bc.location_for(0).has_value());
}

TEST(BytecodeLocationFor, ReturnsNulloptWhenQueryPrecedesFirstEntry)
{
	auto bc = make_bytecode_with_locations({
		InstructionSourceLocation{ /*instruction_index=*/5, /*line=*/10, /*column=*/2 },
	});
	EXPECT_FALSE(bc.location_for(0).has_value());
}

TEST(BytecodeLocationFor, ReturnsExactMatchEntry)
{
	auto bc = make_bytecode_with_locations({
		InstructionSourceLocation{ 0, 1, 0 },
		InstructionSourceLocation{ 3, 7, 4 },
		InstructionSourceLocation{ 10, 12, 0 },
	});
	const auto loc = bc.location_for(3);
	ASSERT_TRUE(loc.has_value());
	EXPECT_EQ(loc->line, 7u);
	EXPECT_EQ(loc->column, 4u);
}

TEST(BytecodeLocationFor, ExtendsEntryUntilNextOne)
{
	auto bc = make_bytecode_with_locations({
		InstructionSourceLocation{ 0, 1, 0 },
		InstructionSourceLocation{ 3, 7, 4 },
		InstructionSourceLocation{ 10, 12, 0 },
	});
	// Query between entries should return the most recent preceding entry.
	for (uint32_t idx : { 0u, 1u, 2u }) {
		const auto loc = bc.location_for(idx);
		ASSERT_TRUE(loc.has_value()) << "idx=" << idx;
		EXPECT_EQ(loc->line, 1u) << "idx=" << idx;
		EXPECT_EQ(loc->column, 0u) << "idx=" << idx;
	}
	for (uint32_t idx : { 3u, 4u, 9u }) {
		const auto loc = bc.location_for(idx);
		ASSERT_TRUE(loc.has_value()) << "idx=" << idx;
		EXPECT_EQ(loc->line, 7u) << "idx=" << idx;
		EXPECT_EQ(loc->column, 4u) << "idx=" << idx;
	}
	for (uint32_t idx : { 10u, 100u, 9999u }) {
		const auto loc = bc.location_for(idx);
		ASSERT_TRUE(loc.has_value()) << "idx=" << idx;
		EXPECT_EQ(loc->line, 12u) << "idx=" << idx;
		EXPECT_EQ(loc->column, 0u) << "idx=" << idx;
	}
}

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
