#include <cxxopts.hpp>

#include "executable/bytecode/BytecodeProgram.hpp"
#include "lexer/Lexer.hpp"
#include "parser/Parser.hpp"
#include "vm/VM.hpp"

#include "Conversion/Passes.hpp"
#include "Conversion/PythonToPythonBytecode/PythonToPythonBytecode.hpp"
#include "Python/MLIRGenerator.hpp"
#include "Target/PythonBytecode/PythonBytecodeEmitter.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <filesystem>

using namespace py;

int main(int argc, char **argv)
{
	cxxopts::Options options("test_mlir", "The C++ Python interpreter MLIR generator");

	// clang-format off
	options.add_options()
		("f,filename", "Script path", cxxopts::value<std::string>())
		("gc-frequency",
		 "Frequency at which the garbage collector is run. Unit is number of allocations",
		 cxxopts::value<uint64_t>()->default_value("10000"))
		("h,help", "Print usage");
	options
		.positional_help("[optional args]")
		.show_positional_help();
	// clang-format on

	options.parse_positional({ "filename" });

	auto result = options.parse(argc, argv);
	if (result.count("help")) {
		std::cout << options.help() << std::endl;
		return EXIT_SUCCESS;
	}

	if (result.count("filename")) {
		size_t arg_idx{ 1 };
		std::vector<std::string> argv_vector;
		argv_vector.reserve(argc - 1);
		while (arg_idx < argc) { argv_vector.emplace_back(argv[arg_idx++]); }

		const auto &filename = result["filename"];
		auto &vm = VirtualMachine::the();
		const auto gc_frequency = result["gc-frequency"].as<uint64_t>();
		vm.heap().garbage_collector().set_frequency(gc_frequency);

		auto lexer = Lexer::create(std::filesystem::absolute(filename.as<std::string>()));
		parser::Parser p{ lexer };
		p.parse();
		// const auto lvl = spdlog::get_level();
		// spdlog::set_level(spdlog::level::debug);
		// p.module()->print_node("");
		// spdlog::set_level(lvl);

		auto ctx = codegen::Context::create();
		if (!codegen::MLIRGenerator::compile(p.module(), std::move(argv_vector), ctx)) {
			std::cerr << "Failed to compile Python script\n";
			return EXIT_FAILURE;
		}
		// ctx.module().dump();

		mlir::py::registerConversionPasses();

		mlir::PassManager pm{ &ctx.ctx() };
		pm.addPass(mlir::py::createPythonToPythonBytecodePass());
		if (pm.run(ctx.module()).failed()) {
			std::cerr << "Python bytecode MLIR lowering failed\n";
			ctx.module().dump();
			return EXIT_FAILURE;
		}

		// ctx.module().dump();

		if (auto program = codegen::translateToPythonBytecode(ctx.module())) {
			// std::cout << program->to_string() << '\n';
			return vm.execute(program);
		}
		std::cerr << "Failed to emit python bytecode from MLIR\n";
		return EXIT_FAILURE;
	}

	std::cout << "Wrong inputs. Usage:\n" << options.help() << std::endl;
	return EXIT_FAILURE;
}