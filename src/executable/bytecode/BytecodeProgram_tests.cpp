#include "BytecodeProgram.hpp"
#include "codegen/BytecodeGenerator.hpp"
#include "lexer/Lexer.hpp"
#include "parser/Parser.hpp"

#include "gtest/gtest.h"

namespace {
std::unique_ptr<BytecodeProgram> generate_bytecode(std::string_view program)
{
	auto lexer = Lexer::create(std::string(program), "_bytecode_program_tests_.py");
	parser::Parser p{ lexer };
	p.parse();

	auto module = as<ast::Module>(p.module());
	ASSERT(module)

	return std::unique_ptr<BytecodeProgram>(static_cast<BytecodeProgram *>(
		codegen::BytecodeGenerator::compile(module, {}, compiler::OptimizationLevel::None)
			.release()));
}
}// namespace

TEST(BytecodeProgramRun, SerializesMainFunction)
{
	static constexpr std::string_view program = "print(\"Hello, world!\")\n";

	auto bytecode_program = generate_bytecode(program);
	auto serialized_bytecode = bytecode_program->serialize();
	ASSERT_TRUE(!serialized_bytecode.empty());
}

TEST(BytecodeProgramRun, DeserializesMainFunction)
{
	static constexpr std::string_view program = "print(\"Hello, world!\")\n";

	auto bytecode_program = generate_bytecode(program);
	auto serialized_bytecode = bytecode_program->serialize();
	auto deserialized_bytecode = BytecodeProgram::deserialize(serialized_bytecode);
	auto executable = std::unique_ptr<Program>(deserialized_bytecode.release());
	ASSERT_EQ(VirtualMachine::the().execute(executable), EXIT_SUCCESS);
}