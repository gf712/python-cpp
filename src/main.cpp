#include "parser/Parser.hpp"
#include "bytecode/BytecodeGenerator.hpp"
#include "bytecode/VM.hpp"

// static constexpr std::string_view program =
// 	"class A:\n"
// 	"	def __init__(self, value):\n"
// 	"		self.value = value\n"
// 	"	def __repr__(self):\n"
// 	"		return \"A\"\n";

static constexpr std::string_view program =
	"a = (1,2,3,4)\n"
	"print(\"values in a:\", 1,2,3,4)\n";

int main()
{
	// spdlog::set_level(spdlog::level::debug);
	Lexer lexer{ std::string(program) };
	parser::Parser p{ lexer };
	p.parse();
	auto bytecode = BytecodeGenerator::compile(p.module());
	spdlog::set_level(spdlog::level::debug);
	spdlog::debug("Bytecode:\n {}", bytecode->to_string());
	auto &vm = VirtualMachine::the().create(std::move(bytecode));
	vm.execute();
	vm.dump();
}
