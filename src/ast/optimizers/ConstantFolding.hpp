#pragma once

#include "ast/AST.hpp"

namespace ast {
namespace optimizer {

	std::shared_ptr<ASTNode> constant_folding(std::shared_ptr<ASTNode> node);

}// namespace optimizer
}// namespace ast