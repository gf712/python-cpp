#include "parser/Parser.hpp"
#include "bytecode/BytecodeGenerator.hpp"
#include "bytecode/VM.hpp"

// static constexpr std::string_view program = "a = 2\n";
// static constexpr std::string_view program =
// 	"a = 2 ** 10\n";

// static constexpr std::string_view program =
// 	"a = 15 + 22 - 1\n"
// 	"b = a\n"
// 	"c = a + b\n";
// static constexpr std::string_view program =
// 	"a = \"foo\"\n"
// 	"b = \"bar\"\n"
// 	"c = \"123\"\n"
// 	"d = a + b + c\n";
// static constexpr std::string_view program =
// 	"a = 2 * 3 + 4 * 5 * 6\n";
// static constexpr std::string_view program = "a = 1 + 2 + 5 * 2 << 10 * 2 + 1\n";
static constexpr std::string_view program =
	"def add(lhs, rhs):\n"
	"	lhs = 1\n"
	"	print(lhs + rhs)\n"
	"	return lhs + rhs\n"
	"a = 3\n"
	"b = 10\n"
	"c = add(a, b)\n";


int main()
{
	spdlog::set_level(spdlog::level::debug);
	Lexer lexer{ std::string(program) };
	parser::Parser p{ lexer };
	p.parse();
	auto bytecode = BytecodeGenerator::compile(p.module());
	spdlog::debug("Bytecode:\n {}", bytecode->to_string());
	auto &vm = VirtualMachine::the().create(std::move(bytecode));
	vm.execute();
	vm.dump();
}
