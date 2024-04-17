#pragma once

namespace compiler {
enum class Backend {
	BYTECODE_GENERATOR = 1,
	LLVM = 2,
	MLIR = 3,
};

enum class OptimizationLevel {
	None = 0,
	Basic = 1,
};
}// namespace compiler

class Program;