#include <cstdio>
#include <cstdlib>
#include "linenoise.h"

#include <iostream>
#include <string>
#include <optional>

#include "bytecode/BytecodeGenerator.hpp"
#include "bytecode/VM.hpp"
#include "interpreter/Interpreter.hpp"
#include "parser/Parser.hpp"

namespace repl {
std::optional<std::string> getline(const std::string &prompt)
{
	if (auto line = linenoise(prompt.c_str())) {
		linenoiseHistoryAdd(line);
		linenoiseHistorySave("history.txt");
		return std::string{ line };
	} else {
		return {};
	}
}


void run(std::string_view program)
{
	static std::shared_ptr<ast::Module> main_module{ nullptr };
	Lexer lexer{ std::string(program) };
	parser::Parser p{ lexer };
	p.parse();
	if (!main_module) {
		main_module = p.module();
	} else {
		for (const auto &node : p.module()->body()) { main_module->emplace(node); }
	}
	auto bytecode = BytecodeGenerator::compile(main_module);
	auto &vm = VirtualMachine::the();
	vm.clear();
	vm.create(std::move(bytecode));
	vm.execute();
}


}// namespace repl

int main(int argc, char** argv)
{
    if (argc > 1) {
        if (strcmp(argv[1], "--debug") == 0) {
        	spdlog::set_level(spdlog::level::debug);
        }
    }
	linenoiseHistoryLoad("history.txt");

	while (auto line = repl::getline("python> ")) {
		(*line) += "\n";
		repl::run(*line);
	}

	return 0;
}