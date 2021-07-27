#include "parser/Parser.hpp"
#include "bytecode/BytecodeGenerator.hpp"
#include "bytecode/VM.hpp"

// static constexpr std::string_view program =
// 	"class A:\n"
// 	"	def __init__(self, value):\n"
// 	"		self.value = value\n"
// 	"	def __repr__(self):\n"
// 	"		return \"A\"\n";

// static constexpr std::string_view program =
// 	"class A:\n"
// 	"	a = 1\n"
// 	"a = A()\n";

static constexpr std::string_view program =
	"a = {\"a\": 1, \"b\":2, \"b\":2}\n";
	// "print(a[\"a\"]\n";

// static constexpr std::string_view program =
// 	"def fibonacci(element):\n"
// 	"	if element == 0:\n"
// 	"		return 0\n"
// 	"	if element == 1:\n"
// 	"		return 1\n"
// 	"	else:\n"
// 	"		lhs = fibonacci(element-1)\n"
// 	"		rhs = fibonacci(element-2)\n"
// 	"		return lhs + rhs\n"
// 	"s1 = fibonacci(15)\n";

// static constexpr std::string_view program =
// 	"a = (1,2,3,4)\n"
// 	"print(\"values in a:\", 1,2,3,4)\n";

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
