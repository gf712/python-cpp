#pragma once

#include "../common.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ast {
    class Module;
}

namespace compiler::mlir {
std::shared_ptr<Program> compile(std::shared_ptr<ast::Module>,
	std::vector<std::string> argv,
	compiler::OptimizationLevel);
}