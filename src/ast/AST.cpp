#include "AST.hpp"

namespace ast {

#define __AST_NODE_TYPE(x)                                                                     \
	template<> std::shared_ptr<x> as(std::shared_ptr<ASTNode> node)                            \
	{                                                                                          \
		if (node->node_type() == ASTNodeType::x) { return std::static_pointer_cast<x>(node); } \
		return nullptr;                                                                        \
	}
AST_NODE_TYPES
#undef __AST_NODE_TYPE

#define __AST_NODE_TYPE(NodeType) \
	void NodeType::codegen(CodeGenerator *generator) const { generator->visit(this); }
AST_NODE_TYPES
#undef __AST_NODE_TYPE

void Constant::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Constant", indent);
	std::visit(overloaded{ [&indent](const String &value) {
							  spdlog::debug("{}  - value: \"{}\"", indent, value.to_string());
						  },
				   [&indent](const auto &value) {
					   spdlog::debug("{}  - value: {}", indent, value.to_string());
				   },
				   [&indent](PyObject *const value) {
					   spdlog::debug("{}  - value: {}", indent, value->to_string());
				   } },
		m_value);
}

void List::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}List", indent);
	spdlog::debug("{}  context: {}", indent, static_cast<int>(m_ctx));
	spdlog::debug("{}  elements:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &el : m_elements) { el->print_node(new_indent); }
}

void Tuple::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Tuple", indent);
	spdlog::debug("{}  context: {}", indent, static_cast<int>(m_ctx));
	spdlog::debug("{}  elements:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &el : m_elements) { el->print_node(new_indent); }
}

void Dict::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Tuple", indent);
	spdlog::debug("{}  keys:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &el : m_keys) { el->print_node(new_indent); }
	spdlog::debug("{}  values:", indent);
	for (const auto &el : m_values) { el->print_node(new_indent); }
}

void Name::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Name", indent);
	spdlog::debug("{}  - id: \"{}\"", indent, m_id[0]);
	spdlog::debug("{}  - context_type: {}", indent, static_cast<int>(m_ctx));
}

void Assign::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Assign", indent);
	spdlog::debug("{}  - targets:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &t : m_targets) { t->print_node(new_indent); }
	spdlog::debug("{}  - value:", indent);
	m_value->print_node(new_indent);
	spdlog::debug("{}  - comment type: {}", indent, m_type_comment);
}

void BinaryExpr::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}BinaryOp", indent);
	spdlog::debug("{}  - op_type: {}", indent, stringify_binary_op(m_op_type));
	spdlog::debug("{}  - lhs:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	m_lhs->print_node(new_indent);
	spdlog::debug("{}  - rhs:", indent);
	m_rhs->print_node(new_indent);
}

void AugAssign::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}AugAssign", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - target:", indent);
	m_target->print_node(new_indent);
	spdlog::debug("{}  - op: {}", indent, stringify_binary_op(m_op));
	spdlog::debug("{}  - value:", indent);
	m_value->print_node(new_indent);
}

void Return::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Return", indent);
	spdlog::debug("{}  - value:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	m_value->print_node(new_indent);
}

void Argument::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Argument", indent);
	spdlog::debug("{}  - arg: {}", indent, m_arg);
	if (m_annotation) {
		spdlog::debug("{}  - annotation:", indent);
		std::string new_indent = indent + std::string(6, ' ');
		m_annotation->print_node(new_indent);
	} else {
		spdlog::debug("{}  - annotation: None", indent);
	}
	spdlog::debug("{}  - type_comment: {}", indent, m_type_comment);
}

std::vector<std::string> Arguments::argument_names() const
{
	std::vector<std::string> arg_names;
	for (const auto &arg : m_args) { arg_names.push_back(arg->name()); }
	return arg_names;
}

void Arguments::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Arguments", indent);
	std::string new_indent = indent + std::string(6, ' ');

	spdlog::debug("{}  - posonlyarg:", indent);
	for (const auto &arg : m_posonlyargs) { arg->print_node(new_indent); }
	spdlog::debug("{}  - args:", indent);
	for (const auto &arg : m_args) { arg->print_node(new_indent); }
	spdlog::debug("{}  - vararg:", indent);
	if (m_vararg) { m_vararg->print_node(new_indent); }
	spdlog::debug("{}  - kwonlyargs:", indent);
	for (const auto &kwarg : m_kwonlyargs) { kwarg->print_node(new_indent); }
	spdlog::debug("{}  - kw_defaults:", indent);
	for (const auto &arg : m_kw_defaults) { arg->print_node(new_indent); }
	spdlog::debug("{}  - kwarg:", indent);
	if (m_kwarg) { m_kwarg->print_node(new_indent); }
	spdlog::debug("{}  - defaults:", indent);
	for (const auto &arg : m_defaults) { arg->print_node(new_indent); }
}

void FunctionDefinition::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}FunctionDefinition", indent);
	spdlog::debug("{}  - function_name: {}", indent, m_function_name);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - args:", indent);
	m_args->print_node(new_indent);
	spdlog::debug("{}  - body:", indent);
	for (const auto &statement : m_body) { statement->print_node(new_indent); }
	spdlog::debug("{}  - decorator_list:", indent);
	for (const auto &decorator : m_decorator_list) { decorator->print_node(new_indent); }
	spdlog::debug("{}  - returns:", indent);
	if (m_returns) m_returns->print_node(new_indent);
	spdlog::debug("{}  - type_comment:{}", indent, m_type_comment);
}

void Keyword::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Keyword", indent);
	if (m_arg.has_value()) {
		spdlog::debug("{}  - arg: {}", indent, *m_arg);
	} else {
		spdlog::debug("{}  - arg: null", indent);
	}
	spdlog::debug("{}  - value:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	m_value->print_node(new_indent);
}

void ClassDefinition::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}ClassDefinition", indent);
	spdlog::debug("{}  - function_name: {}", indent, m_class_name);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - bases:", indent);
	for (const auto &base : m_bases) { base->print_node(new_indent); }
	spdlog::debug("{}  - keywords:", indent);
	for (const auto &keyword : m_keywords) { keyword->print_node(new_indent); }
	spdlog::debug("{}  - body:", indent);
	for (const auto &statement : m_body) { statement->print_node(new_indent); }
	spdlog::debug("{}  - decorator_list:", indent);
	for (const auto &decorator : m_decorator_list) { decorator->print_node(new_indent); }
}

void Call::print_this_node(const std::string &indent) const
{
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}Call", indent);
	spdlog::debug("{}  - function:", indent);
	m_function->print_node(new_indent);
	spdlog::debug("{}  - args:", indent);
	for (const auto &arg : m_args) { arg->print_node(new_indent); }
	spdlog::debug("{}  - keywords:", indent);
	for (const auto &keyword : m_keywords) { keyword->print_node(new_indent); }
}

void Module::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Module", indent);
	spdlog::debug("{}  - body:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &el : m_body) { el->print_node(new_indent); }
}

void If::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}If", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - test:", indent);
	m_test->print_node(new_indent);
	spdlog::debug("{}  - body:", indent);
	for (const auto &el : m_body) { el->print_node(new_indent); }
	spdlog::debug("{}  - orelse:", indent);
	for (const auto &el : m_orelse) { el->print_node(new_indent); }
}

void For::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}For", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - target:", indent);
	m_target->print_node(new_indent);
	spdlog::debug("{}  - iter:", indent);
	m_iter->print_node(new_indent);
	spdlog::debug("{}  - body:", indent);
	for (const auto &el : m_body) { el->print_node(new_indent); }
	spdlog::debug("{}  - orelse:", indent);
	for (const auto &el : m_orelse) { el->print_node(new_indent); }
	spdlog::debug("{}  - type_comment:", m_type_comment);
}

void While::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}While", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - target:", indent);
	m_test->print_node(new_indent);
	spdlog::debug("{}  - body:", indent);
	for (const auto &el : m_body) { el->print_node(new_indent); }
	spdlog::debug("{}  - orelse:", indent);
}

void Compare::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Compare", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - lhs:", indent);
	m_lhs->print_node(new_indent);
	spdlog::debug("{}  - op: {}", indent, op_type_to_string(m_op));
	spdlog::debug("{}  - rhs:", indent);
	m_rhs->print_node(new_indent);
}

void Attribute::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Attribute", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - value:", indent);
	m_value->print_node(new_indent);
	spdlog::debug("{}  - attr: \"{}\"", indent, m_attr);
	spdlog::debug("{}  - ctx: {}", indent, static_cast<int>(m_ctx));
}

void Import::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Import", indent);
	std::string new_indent = indent + std::string(6, ' ');
	if (m_asname.has_value()) {
		spdlog::debug("{}  - asname: \"{}\"", indent, *m_asname);
	} else {
		spdlog::debug("{}  - asname: null", indent);
	}
	spdlog::debug("{}  - name: {}", indent, dotted_name());
}

void Subscript::Index::print(const std::string &indent) const
{
	spdlog::debug("{}Index", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - value:", indent);
	value->print_node(new_indent);
}

void Subscript::Slice::print(const std::string &indent) const
{
	spdlog::debug("{}Slice", indent);
	std::string new_indent = indent + std::string(6, ' ');

	if (lower) {
		spdlog::debug("{}  - lower:", indent);
		lower->print_node(new_indent);
	} else {
		spdlog::debug("{}  - lower: null", indent);
	}

	if (upper) {
		spdlog::debug("{}  - upper:", indent);
		upper->print_node(new_indent);
	} else {
		spdlog::debug("{}  - upper: null", indent);
	}

	if (step) {
		spdlog::debug("{}  - step:", indent);
		step->print_node(new_indent);
	} else {
		spdlog::debug("{}  - step: null", indent);
	}
}

void Subscript::ExtSlice::print(const std::string &indent) const
{
	spdlog::debug("{}ExtSlice", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &d : dims) {
		std::visit([&new_indent](const auto &val) { val.print(new_indent); }, d);
	}
}

void Subscript::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Subscript", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - value:", indent);
	if (m_value) { m_value->print_node(new_indent); }
	spdlog::debug("{}  - slice:", indent);
	if (m_slice) {
		std::visit([&new_indent](const auto &val) { val.print(new_indent); }, *m_slice);
	}
	spdlog::debug("{}  - ctx: {}", indent, m_ctx);
}

void Raise::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Raise", indent);
	std::string new_indent = indent + std::string(6, ' ');
	if (m_exception) {
		spdlog::debug("{}  - exception:", indent);
		m_exception->print_node(new_indent);
	} else {
		spdlog::debug("{}  - exception: null", indent);
	}
	if (m_cause) {
		spdlog::debug("{}  - cause:", indent);
		m_cause->print_node(new_indent);
	} else {
		spdlog::debug("{}  - cause: null", indent);
	}
}

void ExceptHandler::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}ExceptHandler", indent);
	std::string new_indent = indent + std::string(6, ' ');
	if (m_type) {
		spdlog::debug("{}  - type:", indent);
		m_type->print_node(new_indent);
	} else {
		spdlog::debug("{}  - type: null", indent);
	}
	spdlog::debug("{}  - name: {}", indent, m_name);
	spdlog::debug("{}  - body:", indent);
	for (const auto &node : m_body) { node->print_node(new_indent); }
}

void Try::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Try", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - body:", indent);
	for (const auto &node : m_body) { node->print_node(new_indent); }
	spdlog::debug("{}  - handlers:", indent);
	for (const auto &node : m_handlers) { node->print_node(new_indent); }
	spdlog::debug("{}  - orelse:", indent);
	for (const auto &node : m_orelse) { node->print_node(new_indent); }
	spdlog::debug("{}  - finalbody:", indent);
	for (const auto &node : m_finalbody) { node->print_node(new_indent); }
}

void Assert::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Assert", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - test:", indent);
	m_test->print_node(new_indent);
	if (m_msg) {
		spdlog::debug("{}  - message:", indent);
		m_msg->print_node(new_indent);
	} else {
		spdlog::debug("{}  - message: null", indent);
	}
}

void UnaryExpr::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}UnaryExpr", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - op_type: {}", indent, stringify_unary_op(m_op_type));
	m_operand->print_node(new_indent);
}

void BoolOp::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}BoolOp", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - op_type: {}", indent, op_type_to_string(m_op));
	spdlog::debug("{}Values:", indent);
	for (const auto &value : m_values) { value->print_node(new_indent); }
}

void Pass::print_this_node(const std::string &indent) const { spdlog::debug("{}Pass", indent); }

void Global::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Globals", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &name : m_names) { spdlog::debug("{} {}", new_indent, name); }
}

void Delete::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Delete", indent);
	spdlog::debug("{}  - targets", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &target : m_targets) { target->print_node(new_indent); }
}

void With::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}With", indent);
	spdlog::debug("{}  - items", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &item : m_items) { item->print_node(new_indent); }
	for (const auto &statement : m_body) { statement->print_node(new_indent); }
	spdlog::debug("{}  - type_comment: ", indent, m_type_comment);
}

void IfExpr::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}IfExpr", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - test", indent);
	m_test->print_node(new_indent);
	spdlog::debug("{}  - body", indent);
	m_body->print_node(new_indent);
	spdlog::debug("{}  - orelse", indent);
	m_orelse->print_node(new_indent);
}

void Starred::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Starred", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - value", indent);
	m_value->print_node(new_indent);
	spdlog::debug("{}  - context: ", indent, m_ctx);
}

void NamedExpr::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}NamedExpr", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - target", indent);
	m_target->print_node(new_indent);
	spdlog::debug("{}  - value: ", indent);
	m_value->print_node(new_indent);
}

}// namespace ast