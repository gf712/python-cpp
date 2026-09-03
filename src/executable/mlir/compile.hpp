#pragma once

// Include after `import py.runtime;`. Ordinary TUs must include
// executable/common.hpp themselves, ahead of the import.


namespace compiler::mlir {
std::shared_ptr<Program> compile(std::shared_ptr<ast::Module>,
	std::vector<std::string> argv,
	compiler::OptimizationLevel);
}