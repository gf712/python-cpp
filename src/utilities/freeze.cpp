#include "ast/optimizers/ConstantFolding.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/Bytecode.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "executable/bytecode/codegen/BytecodeGenerator.hpp"
#include "parser/Parser.hpp"
#include "runtime/types/builtin.hpp"
#include "vm/VM.hpp"

#include <cxxopts.hpp>

#include <filesystem>
#include <fstream>

using namespace py;

void initialize_types()
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();

	type();
	bool_();
	bytes();
	ellipsis();
	str();
	float_();
	integer();
	none();
	module();
	object();
	dict();
	dict_items();
	dict_items_iterator();
	list();
	list_iterator();
	tuple();
	tuple_iterator();
	range();
	range_iterator();
	function();
	native_function();
	code();
	cell();
	builtin_method();
	slot_wrapper();
	bound_method();
	method_wrapper();
	static_method();
	property();
	classmethod();
}

int freeze(int argc, char **argv, const std::string &output)
{
	size_t arg_idx{ 1 };
	const char *filename = argv[arg_idx];
	std::vector<std::string> argv_vector;
	argv_vector.reserve(argc - 1);
	while (arg_idx < argc) { argv_vector.emplace_back(argv[arg_idx++]); }

	auto &vm = VirtualMachine::the();
	auto lexer = Lexer::create(std::filesystem::absolute(filename));
	parser::Parser p{ lexer };
	p.parse();
	initialize_types();
	auto bytecode = codegen::BytecodeGenerator::compile(
		p.module(), argv_vector, compiler::OptimizationLevel::None);

	std::cout << bytecode->to_string() << "-----------------------------\n\n";

	const auto bytes = bytecode->serialize();

	std::ofstream out;
	out.open(output);
	out << "std::vector<uint8_t> " << std::filesystem::path(filename).stem().c_str() << '\n';
	out << "{\n    ";
	for (size_t idx = 0; const auto &b : bytes) {
		out << static_cast<int>(b) << ", ";
		idx++;
		if (idx == 10) {
			out << "\n    ";
			idx = 0;
		}
	}
	out << "\n};\n";
	out.close();
	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	cxxopts::Options options("freeze", "Freeze a Python file");

	// clang-format off
	options.add_options()
		("f,filename", "Script path", cxxopts::value<std::string>())
		("o,output", "Output header path", cxxopts::value<std::string>())
		("d,debug", "Enable debug logging", cxxopts::value<bool>()->default_value("false"))
		("trace", "Enable trace logging", cxxopts::value<bool>()->default_value("false"))
		("h,help", "Print usage");

    options.parse_positional({ "filename" });

	auto result = options.parse(argc, argv);

	if (result.count("help")) {
		std::cout << options.help() << std::endl;
		return EXIT_SUCCESS;
	}

	const bool debug = result["debug"].as<bool>();
	const bool trace = result["trace"].as<bool>();

	if (result.count("filename")) {
        return freeze(argc, argv, result["output"].as<std::string>());
    } else {
        std::cout << options.help() << std::endl;
		return EXIT_FAILURE;
    }
}