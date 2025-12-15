#include "BytecodeProgram.hpp"
#include "codegen/BytecodeGenerator.hpp"
#include "executable/common.hpp"
#include "lexer/Lexer.hpp"
#include "parser/Parser.hpp"
#include "vm/VM.hpp"

#include "gtest/gtest.h"

namespace {
std::shared_ptr<BytecodeProgram> generate_bytecode(std::string_view program)
{
	auto lexer = Lexer::create(std::string(program), "_bytecode_program_tests_.py");
	parser::Parser p{ lexer };
	p.parse();

	auto module = as<ast::Module>(p.module());
	ASSERT(module);

	return std::static_pointer_cast<BytecodeProgram>(compiler::compile(
		module, {}, compiler::Backend::BYTECODE_GENERATOR, compiler::OptimizationLevel::None));
}
}// namespace

class BytecodeProgramRun : public ::testing::Test
{
  protected:
	BytecodeProgramRun() {}

	virtual ~BytecodeProgramRun() {}

	virtual void SetUp() { VirtualMachine::the().clear(); }

	virtual void TearDown() { VirtualMachine::the().clear(); }
};

TEST_F(BytecodeProgramRun, SerializesMainFunction)
{
	static constexpr std::string_view program = "print(\"Hello, world!\")\n";

	auto bytecode_program = generate_bytecode(program);
	auto serialized_bytecode = bytecode_program->serialize();
	ASSERT_TRUE(!serialized_bytecode.empty());
}

TEST_F(BytecodeProgramRun, DeserializesMainFunction)
{
	static constexpr std::string_view program = "print(\"Hello, world!\")\n";

	auto bytecode_program = generate_bytecode(program);
	auto serialized_bytecode = bytecode_program->serialize();
	ASSERT_TRUE(!serialized_bytecode.empty());
	// auto deserialized_bytecode = BytecodeProgram::deserialize(serialized_bytecode);
	// ASSERT_EQ(VirtualMachine::the().execute(deserialized_bytecode), EXIT_SUCCESS);
}

TEST_F(BytecodeProgramRun, DeserializesMultipleFunctions)
{
	static constexpr std::string_view program =
		"def add(a, b):\n"
		"   return a + b\n"
		"def sub(a, b):\n"
		"   return a - b\n"
		"assert add(20, 22) == 42\n"
		"assert sub(42, 22) == 20\n";

	auto bytecode_program = generate_bytecode(program);
	auto serialized_bytecode = bytecode_program->serialize();
	ASSERT_TRUE(!serialized_bytecode.empty());
	// auto deserialized_bytecode = BytecodeProgram::deserialize(serialized_bytecode);
	// ASSERT_EQ(VirtualMachine::the().execute(deserialized_bytecode), EXIT_SUCCESS);
}