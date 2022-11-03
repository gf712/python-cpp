#include "ConstantFolding.hpp"
#include "ast/AST.hpp"
#include "parser/Parser.hpp"
#include "runtime/Value.hpp"
#include "utilities.hpp"

#include "gtest/gtest.h"

using namespace ast;
using namespace py;

namespace {

void dispatch(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected);

void compare_constant(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Constant);
	const auto result_value = as<Constant>(result)->value();
	const auto expected_value = as<Constant>(expected)->value();

	ASSERT_EQ(result_value->index(), expected_value->index());
	std::visit(
		overloaded{ [&](const Number &number_value) {
					   if (auto *int_result = std::get_if<int64_t>(&number_value.value)) {
						   ASSERT_EQ(*int_result,
							   std::get<int64_t>(std::get<Number>(*expected_value).value));
					   } else if (auto *double_result = std::get_if<double>(&number_value.value)) {
						   ASSERT_EQ(*double_result,
							   std::get<double>(std::get<Number>(*expected_value).value));
					   } else {
						   TODO();
					   }
				   },
			[&](const String &string_value) {
				ASSERT_EQ(string_value.s, std::get<String>(*expected_value).s);
			},
			[&](const NameConstant &name_constant_value) {
				if (auto *bool_result = std::get_if<bool>(&name_constant_value.value)) {
					ASSERT_EQ(*bool_result,
						std::get<bool>(std::get<NameConstant>(*expected_value).value));
				} else {
					TODO();
				}
			},
			[&](const auto &val) {
				(void)val;
				TODO();
				// ASSERT_EQ(result_, std::get<result_.index()>(expected_));
			} },
		*result_value);
}

void compare_assign(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Assign);

	const auto result_targets = as<Assign>(result)->targets();
	const auto expected_targets = as<Assign>(expected)->targets();

	ASSERT_EQ(result_targets.size(), expected_targets.size());
	for (size_t i = 0; i < expected_targets.size(); ++i) {
		dispatch(result_targets[i], expected_targets[i]);
	}

	const auto result_value = as<Assign>(result)->value();
	const auto expected_value = as<Assign>(expected)->value();

	dispatch(result_value, expected_value);
}

void compare_name(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Name);

	const auto result_ids = as<Name>(result)->ids();
	const auto expected_ids = as<Name>(expected)->ids();
	ASSERT_EQ(result_ids.size(), expected_ids.size());
	for (size_t i = 0; i < expected_ids.size(); ++i) { ASSERT_EQ(result_ids[i], expected_ids[i]); }

	ASSERT_EQ(as<Name>(result)->context_type(), as<Name>(expected)->context_type());
}

void compare_binary_expr(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::BinaryExpr);

	const auto result_lhs = as<BinaryExpr>(result)->lhs();
	const auto expected_lhs = as<BinaryExpr>(expected)->lhs();
	dispatch(result_lhs, expected_lhs);

	const auto result_rhs = as<BinaryExpr>(result)->rhs();
	const auto expected_rhs = as<BinaryExpr>(expected)->rhs();
	dispatch(result_rhs, expected_rhs);

	const auto result_optype = as<BinaryExpr>(result)->op_type();
	const auto expected_optype = as<BinaryExpr>(expected)->op_type();

	ASSERT_EQ(result_optype, expected_optype);
}


void compare_function_definition(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::FunctionDefinition);

	const auto result_name = as<FunctionDefinition>(result)->name();
	const auto expected_name = as<FunctionDefinition>(expected)->name();
	EXPECT_EQ(result_name, expected_name);

	// const auto result_args = as<FunctionDefinition>(result)->args();
	// const auto expected_args = as<FunctionDefinition>(expected)->args();
	// dispatch(result_args, expected_args);

	const auto result_body = as<FunctionDefinition>(result)->body();
	const auto expected_body = as<FunctionDefinition>(expected)->body();
	ASSERT_EQ(result_body.size(), expected_body.size());
	for (size_t i = 0; i < result_body.size(); ++i) { dispatch(result_body[i], expected_body[i]); }

	const auto result_decorator_list = as<FunctionDefinition>(result)->decorator_list();
	const auto expected_decorator_list = as<FunctionDefinition>(expected)->decorator_list();
	ASSERT_EQ(result_decorator_list.size(), expected_decorator_list.size());
	for (size_t i = 0; i < result_decorator_list.size(); ++i) {
		dispatch(result_decorator_list[i], expected_decorator_list[i]);
	}

	const auto result_returns = as<FunctionDefinition>(result)->returns();
	const auto expected_returns = as<FunctionDefinition>(expected)->returns();
	dispatch(result_returns, expected_returns);

	const auto result_type_comment = as<FunctionDefinition>(result)->type_comment();
	const auto expected_type_comment = as<FunctionDefinition>(expected)->type_comment();
	EXPECT_EQ(result_type_comment, expected_type_comment);
}


void compare_class_definition(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::ClassDefinition);

	const auto result_name = as<ClassDefinition>(result)->name();
	const auto expected_name = as<ClassDefinition>(expected)->name();
	EXPECT_EQ(result_name, expected_name);

	// const auto result_args = as<FunctionDefinition>(result)->args();
	// const auto expected_args = as<FunctionDefinition>(expected)->args();
	// dispatch(result_args, expected_args);

	const auto result_body = as<ClassDefinition>(result)->body();
	const auto expected_body = as<ClassDefinition>(expected)->body();
	ASSERT_EQ(result_body.size(), expected_body.size());
	for (size_t i = 0; i < result_body.size(); ++i) { dispatch(result_body[i], expected_body[i]); }

	const auto result_decorator_list = as<ClassDefinition>(result)->decorator_list();
	const auto expected_decorator_list = as<ClassDefinition>(expected)->decorator_list();
	ASSERT_EQ(result_decorator_list.size(), expected_decorator_list.size());
	for (size_t i = 0; i < result_decorator_list.size(); ++i) {
		dispatch(result_decorator_list[i], expected_decorator_list[i]);
	}
}


void compare_return(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Return);

	const auto result_value = as<Return>(result)->value();
	const auto expected_value = as<Return>(expected)->value();
	dispatch(result_value, expected_value);
}

void compare_if(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::If);

	const auto result_test = as<If>(result)->test();
	const auto expected_test = as<If>(expected)->test();
	dispatch(result_test, expected_test);

	const auto result_body = as<If>(result)->body();
	const auto expected_body = as<If>(expected)->body();
	ASSERT_EQ(result_body.size(), expected_body.size());
	for (size_t i = 0; i < expected_body.size(); ++i) {
		dispatch(result_body[i], expected_body[i]);
	}

	const auto result_orelse = as<If>(result)->orelse();
	const auto expected_orelse = as<If>(expected)->orelse();
	ASSERT_EQ(result_orelse.size(), expected_orelse.size());
	for (size_t i = 0; i < result_orelse.size(); ++i) {
		dispatch(result_orelse[i], expected_orelse[i]);
	}
}

void compare_for(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::For);

	const auto result_target = as<For>(result)->target();
	const auto expected_target = as<For>(expected)->target();
	dispatch(result_target, expected_target);

	const auto result_iter = as<For>(result)->iter();
	const auto expected_iter = as<For>(expected)->iter();
	dispatch(result_iter, expected_iter);

	const auto result_body = as<For>(result)->body();
	const auto expected_body = as<For>(expected)->body();
	ASSERT_EQ(result_body.size(), expected_body.size());
	for (size_t i = 0; i < expected_body.size(); ++i) {
		dispatch(result_body[i], expected_body[i]);
	}

	const auto result_orelse = as<For>(result)->orelse();
	const auto expected_orelse = as<For>(expected)->orelse();
	ASSERT_EQ(result_orelse.size(), expected_orelse.size());
	for (size_t i = 0; i < result_orelse.size(); ++i) {
		dispatch(result_orelse[i], expected_orelse[i]);
	}

	const auto result_type_comment = as<For>(result)->type_comment();
	const auto expected_type_comment = as<For>(expected)->type_comment();
	ASSERT_EQ(result_type_comment, expected_type_comment);
}

void compare_call(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Call);

	const auto result_function = as<Call>(result)->function();
	const auto expected_function = as<Call>(expected)->function();
	dispatch(result_function, expected_function);

	const auto result_args = as<Call>(result)->args();
	const auto expected_args = as<Call>(expected)->args();
	ASSERT_EQ(result_args.size(), expected_args.size());
	for (size_t i = 0; i < expected_args.size(); ++i) {
		dispatch(result_args[i], expected_args[i]);
	}

	const auto result_keywords = as<Call>(result)->keywords();
	const auto expected_keywords = as<Call>(expected)->keywords();
	ASSERT_EQ(result_keywords.size(), expected_keywords.size());
	for (size_t i = 0; i < result_keywords.size(); ++i) {
		dispatch(result_keywords[i], expected_keywords[i]);
	}
}

void compare_compare(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Compare);

	const auto result_lhs = as<Compare>(result)->lhs();
	const auto expected_lhs = as<Compare>(expected)->lhs();
	dispatch(result_lhs, expected_lhs);

	ASSERT_EQ(as<Compare>(result)->ops().size(), as<Compare>(expected)->ops().size());
	for (size_t i = 0; i < as<Compare>(result)->ops().size(); ++i) {
		const auto result_op = as<Compare>(result)->ops()[i];
		const auto expected_op = as<Compare>(expected)->ops()[i];
		ASSERT_EQ(result_op, expected_op);
	}

	ASSERT_EQ(
		as<Compare>(result)->comparators().size(), as<Compare>(expected)->comparators().size());
	for (size_t i = 0; i < as<Compare>(result)->comparators().size(); ++i) {
		const auto result_cmp = as<Compare>(result)->comparators()[i];
		const auto expected_cmp = as<Compare>(expected)->comparators()[i];
		dispatch(result_cmp, expected_cmp);
	}
}


void compare_list(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::List);

	const auto result_context = as<List>(result)->context();
	const auto expected_context = as<List>(expected)->context();
	ASSERT_EQ(result_context, expected_context);

	const auto result_elements = as<List>(result)->elements();
	const auto expected_elements = as<List>(expected)->elements();
	ASSERT_EQ(result_elements.size(), expected_elements.size());
	for (size_t i = 0; i < result_elements.size(); ++i) {
		dispatch(expected_elements[i], expected_elements[i]);
	}
}

void compare_tuple(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Tuple);

	const auto result_context = as<Tuple>(result)->context();
	const auto expected_context = as<Tuple>(expected)->context();
	ASSERT_EQ(result_context, expected_context);

	const auto result_elements = as<Tuple>(result)->elements();
	const auto expected_elements = as<Tuple>(expected)->elements();
	ASSERT_EQ(result_elements.size(), expected_elements.size());
	for (size_t i = 0; i < result_elements.size(); ++i) {
		dispatch(expected_elements[i], expected_elements[i]);
	}
}

void compare_dict(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Dict);

	const auto result_values = as<Dict>(result)->values();
	const auto expected_values = as<Dict>(expected)->values();
	ASSERT_EQ(result_values.size(), expected_values.size());
	for (size_t i = 0; i < result_values.size(); ++i) {
		dispatch(result_values[i], expected_values[i]);
	}

	const auto result_keys = as<Dict>(result)->keys();
	const auto expected_keys = as<Dict>(expected)->keys();
	ASSERT_EQ(result_keys.size(), expected_keys.size());
	for (size_t i = 0; i < result_keys.size(); ++i) { dispatch(result_keys[i], expected_keys[i]); }
}

void compare_attribute(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Attribute);

	const auto result_context = as<Attribute>(result)->context();
	const auto expected_context = as<Attribute>(expected)->context();
	ASSERT_EQ(result_context, expected_context);

	const auto result_attr = as<Attribute>(result)->attr();
	const auto expected_attr = as<Attribute>(expected)->attr();
	ASSERT_EQ(result_attr, expected_attr);

	const auto result_value = as<Attribute>(result)->value();
	const auto expected_value = as<Attribute>(expected)->value();
	dispatch(result_value, expected_value);
}

void compare_keyword(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Keyword);

	const auto result_arg = as<Keyword>(result)->arg();
	const auto expected_arg = as<Keyword>(expected)->arg();
	ASSERT_EQ(result_arg, expected_arg);

	const auto result_value = as<Keyword>(result)->value();
	const auto expected_value = as<Keyword>(expected)->value();
	dispatch(result_value, expected_value);
}

void compare_augmented_assign(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::AugAssign);

	const auto result_target = as<AugAssign>(result)->target();
	const auto expected_target = as<AugAssign>(expected)->target();
	dispatch(result_target, expected_target);

	const auto result_op = as<AugAssign>(result)->op();
	const auto expected_op = as<AugAssign>(expected)->op();
	ASSERT_EQ(result_op, expected_op);

	const auto result_value = as<AugAssign>(result)->value();
	const auto expected_value = as<AugAssign>(expected)->value();
	dispatch(result_value, expected_value);
}


void dispatch(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	if (!expected) {
		ASSERT_FALSE(result);
		return;
	}
	switch (expected->node_type()) {
	case ASTNodeType::Constant: {
		compare_constant(result, expected);
		break;
	}
	case ASTNodeType::Assign: {
		compare_assign(result, expected);
		break;
	}
	case ASTNodeType::Name: {
		compare_name(result, expected);
		break;
	}
	case ASTNodeType::BinaryExpr: {
		compare_binary_expr(result, expected);
		break;
	}
	case ASTNodeType::FunctionDefinition: {
		compare_function_definition(result, expected);
		break;
	}
	case ASTNodeType::Return: {
		compare_return(result, expected);
		break;
	}
	case ASTNodeType::If: {
		compare_if(result, expected);
		break;
	}
	case ASTNodeType::Call: {
		compare_call(result, expected);
		break;
	}
	case ASTNodeType::Compare: {
		compare_compare(result, expected);
		break;
	}
	case ASTNodeType::List: {
		compare_list(result, expected);
		break;
	}
	case ASTNodeType::Tuple: {
		compare_tuple(result, expected);
		break;
	}
	case ASTNodeType::Dict: {
		compare_dict(result, expected);
		break;
	}
	case ASTNodeType::For: {
		compare_for(result, expected);
		break;
	}
	case ASTNodeType::ClassDefinition: {
		compare_class_definition(result, expected);
		break;
	}
	case ASTNodeType::Attribute: {
		compare_attribute(result, expected);
		break;
	}
	case ASTNodeType::Keyword: {
		compare_keyword(result, expected);
		break;
	}
	case ASTNodeType::AugAssign: {
		compare_augmented_assign(result, expected);
		break;
	}
	default: {
		spdlog::error("Unhandled AST node type {}", node_type_to_string(expected->node_type()));
		TODO();
	}
	}
}

void assert_generates_ast(std::string_view program,
	std::shared_ptr<Module> expected_module,
	compiler::OptimizationLevel lvl)
{
	auto lexer = Lexer::create(std::string(program), "_optimizers_dummy_.py");
	parser::Parser p{ lexer };
	const auto spdlog_level = spdlog::get_level();
	spdlog::set_level(spdlog::level::debug);
	p.parse();
	spdlog::set_level(spdlog_level);

	if (lvl > compiler::OptimizationLevel::None) {
		auto module = p.module();
		ast::optimizer::constant_folding(module);
	}

	size_t i = 0;
	for (const auto &node : p.module()->body()) {
		dispatch(node, expected_module->body()[i]);
		i++;
	}

	spdlog::set_level(spdlog::level::debug);
	p.module()->print_node("");
	expected_module->print_node("");
	spdlog::set_level(spdlog_level);
}

std::shared_ptr<Module> create_test_module()
{
	return std::make_shared<Module>("_optimizers_dummy_.py");
}
}// namespace

TEST(Optimizer, ConstantFoldIntegerAddition)
{
	constexpr std::string_view program = "a = 1 + 1\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Constant>(static_cast<int64_t>(2), SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast, compiler::OptimizationLevel::Basic);
}
