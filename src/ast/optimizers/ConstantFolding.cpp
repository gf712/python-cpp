#include "ConstantFolding.hpp"
#include "runtime/Value.hpp"

namespace ast {
namespace optimizer {

	std::variant<py::Value, std::shared_ptr<BinaryExpr>> evaluate_binary_expr(
		const std::shared_ptr<BinaryExpr> &node)
	{
		if (node->lhs()->node_type() == ASTNodeType::Constant
			&& node->rhs()->node_type() == ASTNodeType::Constant) {
			const auto &lhs = as<Constant>(node->lhs())->value();
			const auto &rhs = as<Constant>(node->rhs())->value();
			ASSERT(lhs)
			ASSERT(rhs)
			switch (node->op_type()) {
			case BinaryOpType::PLUS: {
				auto result = std::visit(
					overloaded{ [](const py::Number &lhs_value, const py::Number &rhs_value)
									-> std::optional<py::Value> { return lhs_value + rhs_value; },
						[](const auto &, const auto &) -> std::optional<py::Value> { return {}; } },
					*lhs,
					*rhs);
				if (result) { return *result; }

			} break;
			case BinaryOpType::MINUS: {
				TODO();
			} break;
			case BinaryOpType::MULTIPLY: {
				TODO();
			} break;
			case BinaryOpType::EXP: {
				TODO();
			} break;
			case BinaryOpType::MODULO: {
				TODO();
			} break;
			case BinaryOpType::SLASH: {
				TODO();
			} break;
			case BinaryOpType::FLOORDIV: {
				TODO();
			} break;
			case BinaryOpType::MATMUL: {
				TODO();
			} break;
			case BinaryOpType::LEFTSHIFT: {
				TODO();
			} break;
			case BinaryOpType::RIGHTSHIFT: {
				TODO();
			} break;
			case BinaryOpType::AND: {
				TODO();
			} break;
			case BinaryOpType::OR: {
				TODO();
			} break;
			case BinaryOpType::XOR: {
				TODO();
			} break;
			}
		}

		return node;
	}

	std::shared_ptr<ASTNode> constant_folding(std::shared_ptr<ASTNode> node)
	{
		spdlog::debug("Constant folding optimization");
		if (node->node_type() == ASTNodeType::BinaryExpr) {
			auto result = evaluate_binary_expr(as<BinaryExpr>(node));
			if (std::holds_alternative<py::Value>(result)) {
				spdlog::debug("Evaluated binary node - creating new constant node");
				return std::make_shared<Constant>(
					std::get<py::Value>(result), node->source_location());
			}
		} else if (node->node_type() == ASTNodeType::Constant) {
		} else if (node->node_type() == ASTNodeType::Assign) {
			auto result = constant_folding(as<Assign>(node)->value());
			as<Assign>(node)->set_value(result);
		} else if (node->node_type() == ASTNodeType::Module) {
			for (auto n : as<Module>(node)->body()) { constant_folding(n); }
		}

		return node;
	}
}// namespace optimizer
}// namespace ast
