#pragma once

#include "ast/AST.hpp"
#include "utilities.hpp"

namespace compiler {

enum class OptimizationLevel { None = 0, Basic = 1 };

}

namespace ast {
namespace optimizer {

	std::shared_ptr<ASTNode> constant_folding(std::shared_ptr<ASTNode> node);

}// namespace optimizer
}// namespace ast