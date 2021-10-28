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
	"class A:\n"
	"	a = 1\n"
	"	def __repr__(self):\n"
	"		return self.a\n"
	"	def bar(self):\n"
	"		print(self)\n"
	"foo = A()\n"
	"print(foo)\n"
	"foo.bar()\n";

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
// 	"s1 = fibonacci(25)\n"
// 	"print(s1)\n";

// static constexpr std::string_view program =
// 	"a = (1,2,3,4)\n"
// 	"b = a + a\n"
// 	"print(b)\n";

int main()
{
	spdlog::set_level(spdlog::level::debug);
	Lexer lexer{ std::string(program) };
	parser::Parser p{ lexer };
	p.parse();
	p.module()->print_node("");
	auto bytecode = BytecodeGenerator::compile(p.module(), compiler::OptimizationLevel::None);
	spdlog::debug("Bytecode:\n {}", bytecode->to_string());
	auto &vm = VirtualMachine::the().create(std::move(bytecode));
	// spdlog::set_level(spdlog::level::debug);
	vm.execute();
	// vm.dump();
}