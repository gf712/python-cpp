#include <cstdio>
#include <cstdlib>
#include "linenoise.h"

#include <iostream>
#include <string>
#include <optional>

#include "ast/optimizers/ConstantFolding.hpp"
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


class InteractivePython
{
  public:
	InteractivePython() {}

	std::shared_ptr<PyObject> interpret_statement(std::string statement)
	{
		Lexer lexer{ statement };
		parser::Parser parser{ lexer };
		parser.parse();
		if (!m_main_module) {
			m_main_module = parser.module();
		} else {
			for (const auto &node : parser.module()->body()) { m_main_module->emplace(node); }
		}
		auto bytecode = BytecodeGenerator::compile(m_main_module, compiler::OptimizationLevel::None);
		auto &vm = VirtualMachine::the();
		return vm.execute_statement(bytecode);
	}

  private:
	std::shared_ptr<ast::Module> m_main_module{ nullptr };
};

}// namespace repl

int main(int argc, char **argv)
{
	if (argc > 1) {
		if (strcmp(argv[1], "--debug") == 0) { spdlog::set_level(spdlog::level::debug); }
	}
	linenoiseHistoryLoad("history.txt");

	repl::InteractivePython interactive_interpreter;

	static constexpr std::string_view major_version = "3";
	static constexpr std::string_view minor_version = "10";
	static constexpr std::string_view build_version = "0a";
	static constexpr std::string_view compiler = "Clang 11.0.0";
	static constexpr std::string_view platform = "Linux";

	std::cout << fmt::format("Python {}.{}.{}\n", major_version, minor_version, build_version);
	std::cout << fmt::format("[{}] :: {}\n", compiler, platform);
	std::cout << "Type \"help\", \"copyright\", \"credits\" or \"license\" for more information.";
	std::cout << std::endl;

	while (auto line = repl::getline(">>> ")) {
		(*line) += "\n";
		auto result = interactive_interpreter.interpret_statement(*line);
		if (result.get() != py_none().get()) {
			std::cout << result->repr_impl(*VirtualMachine::the().interpreter())->to_string()
					  << '\n';
		}
	}

	return 0;
}