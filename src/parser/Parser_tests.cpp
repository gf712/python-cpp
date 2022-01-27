#include "ast/AST.hpp"

#include "parser/Parser.hpp"

#include "utilities.hpp"

#include "gtest/gtest.h"

using namespace ast;

namespace {

void dispatch(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected);

void compare_constant(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Constant);
	const auto result_value = as<Constant>(result)->value();
	const auto expected_value = as<Constant>(expected)->value();

	ASSERT_EQ(result_value.index(), expected_value.index());
	std::visit(
		overloaded{ [&](const Number &number_value) {
					   if (auto *int_result = std::get_if<int64_t>(&number_value.value)) {
						   ASSERT_EQ(*int_result,
							   std::get<int64_t>(std::get<Number>(expected_value).value));
					   } else if (auto *double_result = std::get_if<double>(&number_value.value)) {
						   ASSERT_EQ(*double_result,
							   std::get<double>(std::get<Number>(expected_value).value));
					   } else {
						   TODO();
					   }
				   },
			[&](const String &string_value) {
				ASSERT_EQ(string_value.s, std::get<String>(expected_value).s);
			},
			[&](const NameConstant &name_constant_value) {
				if (auto *bool_result = std::get_if<bool>(&name_constant_value.value)) {
					ASSERT_EQ(
						*bool_result, std::get<bool>(std::get<NameConstant>(expected_value).value));
				} else if (std::holds_alternative<NoneType>(name_constant_value.value)) {
					ASSERT_TRUE(std::holds_alternative<NoneType>(name_constant_value.value));
				} else {
					TODO();
				}
			},
			[&](const auto &val) {
				(void)val;
				TODO();
				// ASSERT_EQ(result_, std::get<result_.index()>(expected_));
			} },
		result_value);
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

	const auto result_args = as<FunctionDefinition>(result)->args();
	const auto expected_args = as<FunctionDefinition>(expected)->args();
	dispatch(result_args, expected_args);

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

	const auto result_bases = as<ClassDefinition>(result)->bases();
	const auto expected_bases = as<ClassDefinition>(expected)->bases();
	ASSERT_EQ(result_bases.size(), expected_bases.size());
	for (size_t i = 0; i < result_bases.size(); ++i) {
		dispatch(result_bases[i], expected_bases[i]);
	}

	const auto result_keywords = as<ClassDefinition>(result)->keywords();
	const auto expected_keywords = as<ClassDefinition>(expected)->keywords();
	ASSERT_EQ(result_keywords.size(), expected_keywords.size());
	for (size_t i = 0; i < result_keywords.size(); ++i) {
		dispatch(result_keywords[i], expected_keywords[i]);
	}

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

void compare_while(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::While);

	const auto result_test = as<While>(result)->test();
	const auto expected_test = as<While>(expected)->test();
	dispatch(result_test, expected_test);

	const auto result_body = as<While>(result)->body();
	const auto expected_body = as<While>(expected)->body();
	ASSERT_EQ(result_body.size(), expected_body.size());
	for (size_t i = 0; i < expected_body.size(); ++i) {
		dispatch(result_body[i], expected_body[i]);
	}

	const auto result_orelse = as<While>(result)->orelse();
	const auto expected_orelse = as<While>(expected)->orelse();
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

	const auto result_op = as<Compare>(result)->op();
	const auto expected_op = as<Compare>(expected)->op();
	ASSERT_EQ(result_op, expected_op);

	const auto result_rhs = as<Compare>(result)->rhs();
	const auto expected_rhs = as<Compare>(expected)->rhs();
	dispatch(result_rhs, expected_rhs);
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
	ASSERT_EQ(result_arg.has_value(), expected_arg.has_value());
	if (expected_arg.has_value()) { ASSERT_EQ(*result_arg, *expected_arg); }

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


void compare_import(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Import);

	const auto result_names = as<Import>(result)->names();
	const auto expected_names = as<Import>(expected)->names();
	ASSERT_EQ(result_names.size(), expected_names.size());

	for (size_t i = 0; i < result_names.size(); ++i) {
		ASSERT_EQ(result_names[i], expected_names[i]);
	}

	const auto result_asname = as<Import>(result)->asname();
	const auto expected_asname = as<Import>(expected)->asname();
	ASSERT_EQ(result_asname.has_value(), expected_asname.has_value());

	if (expected_asname.has_value()) { ASSERT_EQ(*result_asname, *expected_asname); }
}


void compare_subscript(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Subscript);

	const auto result_value = as<Subscript>(result)->value();
	const auto expected_value = as<Subscript>(expected)->value();
	dispatch(result_value, expected_value);

	const auto result_slice = as<Subscript>(result)->slice();
	const auto expected_slice = as<Subscript>(expected)->slice();

	ASSERT_EQ(result_slice.index(), expected_slice.index());
	if (std::holds_alternative<Subscript::Slice>(expected_slice)) {
		auto result_s = std::get<Subscript::Slice>(result_slice);
		auto expected_s = std::get<Subscript::Slice>(expected_slice);

		if (expected_s.lower) {
			ASSERT_TRUE(result_s.lower);
			dispatch(result_s.lower, expected_s.lower);
		} else {
			ASSERT_FALSE(result_s.lower);
		}
		if (expected_s.upper) {
			ASSERT_TRUE(result_s.upper);
			dispatch(result_s.upper, expected_s.upper);
		} else {
			ASSERT_FALSE(result_s.upper);
		}
		if (expected_s.step) {
			ASSERT_TRUE(result_s.step);
			dispatch(result_s.step, expected_s.step);
		} else {
			ASSERT_FALSE(result_s.step);
		}
	} else if (std::holds_alternative<Subscript::Index>(expected_slice)) {
		auto result_i = std::get<Subscript::Index>(result_slice);
		auto expected_i = std::get<Subscript::Index>(expected_slice);

		if (expected_i.value) {
			ASSERT_TRUE(result_i.value);
			dispatch(result_i.value, expected_i.value);
		} else {
			ASSERT_FALSE(result_i.value);
		}

	} else {
		TODO();
	}

	const auto result_ctx = as<Subscript>(result)->context();
	const auto expected_ctx = as<Subscript>(expected)->context();
	ASSERT_EQ(result_ctx, expected_ctx);
}

void compare_raise(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Raise);

	const auto result_exception = as<Raise>(result)->exception();
	const auto expected_exception = as<Raise>(expected)->exception();
	dispatch(result_exception, expected_exception);

	const auto result_cause = as<Raise>(result)->cause();
	const auto expected_cause = as<Raise>(expected)->cause();
	dispatch(result_exception, expected_exception);
}

void compare_try(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Try);

	const auto result_body = as<Try>(result)->body();
	const auto expected_body = as<Try>(expected)->body();
	ASSERT_EQ(result_body.size(), expected_body.size());
	for (size_t i = 0; i < result_body.size(); ++i) { dispatch(result_body[i], expected_body[i]); }

	const auto result_handlers = as<Try>(result)->handlers();
	const auto expected_handlers = as<Try>(expected)->handlers();
	ASSERT_EQ(result_handlers.size(), expected_handlers.size());
	for (size_t i = 0; i < result_handlers.size(); ++i) {
		dispatch(result_handlers[i], expected_handlers[i]);
	}

	const auto result_orelse = as<Try>(result)->orelse();
	const auto expected_orelse = as<Try>(expected)->orelse();
	ASSERT_EQ(result_orelse.size(), expected_orelse.size());
	for (size_t i = 0; i < result_orelse.size(); ++i) {
		dispatch(result_orelse[i], expected_orelse[i]);
	}

	const auto result_finalbody = as<Try>(result)->finalbody();
	const auto expected_finalbody = as<Try>(expected)->finalbody();
	ASSERT_EQ(result_finalbody.size(), expected_finalbody.size());
	for (size_t i = 0; i < result_finalbody.size(); ++i) {
		dispatch(result_finalbody[i], expected_finalbody[i]);
	}
}

void compare_except_handler(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::ExceptHandler);

	const auto result_type = as<ExceptHandler>(result)->type();
	const auto expected_type = as<ExceptHandler>(expected)->type();
	dispatch(result_type, expected_type);

	const auto result_name = as<ExceptHandler>(result)->name();
	const auto expected_name = as<ExceptHandler>(expected)->name();
	ASSERT_EQ(result_name.size(), expected_name.size());

	const auto result_body = as<ExceptHandler>(result)->body();
	const auto expected_body = as<ExceptHandler>(expected)->body();
	ASSERT_EQ(result_body.size(), expected_body.size());
	for (size_t i = 0; i < result_body.size(); ++i) { dispatch(result_body[i], expected_body[i]); }
}

void compare_assert(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Assert);

	const auto result_test = as<Assert>(result)->test();
	const auto expected_test = as<Assert>(expected)->test();
	dispatch(result_test, expected_test);

	const auto result_msg = as<Assert>(result)->msg();
	const auto expected_msg = as<Assert>(expected)->msg();
	dispatch(result_msg, expected_msg);
}

void compare_unary_op(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::UnaryExpr);

	const auto result_operand = as<UnaryExpr>(result)->operand();
	const auto expected_operand = as<UnaryExpr>(expected)->operand();
	dispatch(result_operand, expected_operand);

	const auto result_optype = as<UnaryExpr>(result)->op_type();
	const auto expected_optype = as<UnaryExpr>(expected)->op_type();

	ASSERT_EQ(result_optype, expected_optype);
}

void compare_bool_op(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::BoolOp);

	const auto result_op = as<BoolOp>(result)->op();
	const auto expected_op = as<BoolOp>(expected)->op();
	ASSERT_EQ(result_op, expected_op);

	const auto result_values = as<BoolOp>(result)->values();
	const auto expected_values = as<BoolOp>(expected)->values();

	ASSERT_EQ(result_values.size(), expected_values.size());
	for (size_t i = 0; i < result_values.size(); ++i) {
		dispatch(result_values[i], expected_values[i]);
	}
}

void compare_arguments(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Arguments);

	const auto result_posonlyargs = as<Arguments>(result)->posonlyargs();
	const auto expected_posonlyargs = as<Arguments>(expected)->posonlyargs();
	ASSERT_EQ(result_posonlyargs.size(), expected_posonlyargs.size());
	for (size_t i = 0; i < result_posonlyargs.size(); ++i) {
		dispatch(result_posonlyargs[i], expected_posonlyargs[i]);
	}

	const auto result_args = as<Arguments>(result)->args();
	const auto expected_args = as<Arguments>(expected)->args();
	ASSERT_EQ(result_args.size(), expected_args.size());
	for (size_t i = 0; i < result_args.size(); ++i) { dispatch(result_args[i], expected_args[i]); }

	const auto result_kwonlyargs = as<Arguments>(result)->kwonlyargs();
	const auto expected_kwonlyargs = as<Arguments>(expected)->kwonlyargs();
	ASSERT_EQ(result_kwonlyargs.size(), expected_kwonlyargs.size());
	for (size_t i = 0; i < result_kwonlyargs.size(); ++i) {
		dispatch(result_kwonlyargs[i], expected_kwonlyargs[i]);
	}

	const auto result_vararg = as<Arguments>(result)->vararg();
	const auto expected_vararg = as<Arguments>(expected)->vararg();
	dispatch(result_vararg, expected_vararg);

	const auto result_kwarg = as<Arguments>(result)->kwarg();
	const auto expected_kwarg = as<Arguments>(expected)->kwarg();
	dispatch(result_kwarg, expected_kwarg);

	const auto result_kw_defaults = as<Arguments>(result)->kw_defaults();
	const auto expected_kw_defaults = as<Arguments>(expected)->kw_defaults();
	ASSERT_EQ(result_kw_defaults.size(), expected_kw_defaults.size());
	for (size_t i = 0; i < result_kw_defaults.size(); ++i) {
		dispatch(result_kw_defaults[i], expected_kw_defaults[i]);
	}

	const auto result_defaults = as<Arguments>(result)->defaults();
	const auto expected_defaults = as<Arguments>(expected)->defaults();
	ASSERT_EQ(result_defaults.size(), expected_defaults.size());
	for (size_t i = 0; i < result_defaults.size(); ++i) {
		dispatch(result_defaults[i], expected_defaults[i]);
	}
}

void compare_argument(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Argument);

	const auto result_name = as<Argument>(result)->name();
	const auto expected_name = as<Argument>(expected)->name();

	ASSERT_EQ(result_name, expected_name);

	const auto result_annotation = as<Argument>(result)->annotation();
	const auto expected_annotation = as<Argument>(expected)->annotation();

	dispatch(result_annotation, expected_annotation);
}


void compare_with_statement(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::With);

	const auto result_items = as<With>(result)->items();
	const auto expected_items = as<With>(expected)->items();

	ASSERT_EQ(result_items.size(), expected_items.size());

	for (size_t i = 0; i < result_items.size(); ++i) {
		dispatch(result_items[i], expected_items[i]);
	}

	const auto result_body = as<With>(result)->body();
	const auto expected_body = as<With>(expected)->body();

	ASSERT_EQ(result_body.size(), expected_body.size());

	for (size_t i = 0; i < result_body.size(); ++i) { dispatch(result_body[i], expected_body[i]); }

	const auto result_type_comment = as<With>(result)->type_comment();
	const auto expected_type_comment = as<With>(expected)->type_comment();
	ASSERT_EQ(result_type_comment, expected_type_comment);
}

void compare_if_expression(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::IfExpr);

	const auto result_test = as<IfExpr>(result)->test();
	const auto expected_test = as<IfExpr>(expected)->test();
	dispatch(result_test, expected_test);

	const auto result_body = as<IfExpr>(result)->body();
	const auto expected_body = as<IfExpr>(expected)->body();
	dispatch(result_body, expected_body);

	const auto result_orelse = as<IfExpr>(result)->orelse();
	const auto expected_orelse = as<IfExpr>(expected)->orelse();
	dispatch(result_orelse, expected_orelse);
}

void compare_starred(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Starred);

	const auto result_value = as<Starred>(result)->value();
	const auto expected_value = as<Starred>(expected)->value();
	dispatch(result_value, expected_value);

	const auto result_ctx = as<Starred>(result)->ctx();
	const auto expected_ctx = as<Starred>(expected)->ctx();
	ASSERT_EQ(result_ctx, expected_ctx);
}

void compare_pass(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Pass);
}

void compare_named_expression(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::NamedExpr);

	const auto result_target = as<NamedExpr>(result)->target();
	const auto expected_target = as<NamedExpr>(expected)->target();
	dispatch(result_target, expected_target);

	const auto result_value = as<NamedExpr>(result)->value();
	const auto expected_value = as<NamedExpr>(expected)->value();
	dispatch(result_value, expected_value);
}

void dispatch(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	if (!expected) {
		ASSERT_FALSE(result);
		return;
	}
	if (!result) {
		ASSERT_FALSE(expected);
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
	case ASTNodeType::While: {
		compare_while(result, expected);
		break;
	}
	case ASTNodeType::Import: {
		compare_import(result, expected);
		break;
	}
	case ASTNodeType::Subscript: {
		compare_subscript(result, expected);
		break;
	}
	case ASTNodeType::Raise: {
		compare_raise(result, expected);
		break;
	}
	case ASTNodeType::Try: {
		compare_try(result, expected);
		break;
	}
	case ASTNodeType::ExceptHandler: {
		compare_except_handler(result, expected);
		break;
	}
	case ASTNodeType::Assert: {
		compare_assert(result, expected);
		break;
	}
	case ASTNodeType::UnaryExpr: {
		compare_unary_op(result, expected);
		break;
	}
	case ASTNodeType::BoolOp: {
		compare_bool_op(result, expected);
		break;
	}
	case ASTNodeType::Arguments: {
		compare_arguments(result, expected);
		break;
	}
	case ASTNodeType::Argument: {
		compare_argument(result, expected);
		break;
	}
	case ASTNodeType::With: {
		compare_with_statement(result, expected);
		break;
	}
	case ASTNodeType::IfExpr: {
		compare_if_expression(result, expected);
		break;
	}
	case ASTNodeType::Starred: {
		compare_starred(result, expected);
		break;
	}
	case ASTNodeType::Pass: {
		compare_pass(result, expected);
		break;
	}
	case ASTNodeType::NamedExpr: {
		compare_named_expression(result, expected);
		break;
	}
	default: {
		spdlog::error("Unhandled AST node type {}", node_type_to_string(expected->node_type()));
		TODO();
	}
	}
}

void assert_generates_ast(std::string_view program, std::shared_ptr<Module> expected_module)
{
	auto lexer = Lexer::create(std::string(program), "_parser_test_.py");
	parser::Parser p{ lexer };
	// spdlog::set_level(spdlog::level::debug);
	p.parse();
	// spdlog::set_level(spdlog::level::info);

	ASSERT_EQ(p.module()->body().size(), expected_module->body().size());

	size_t i = 0;
	for (const auto &node : p.module()->body()) {
		dispatch(node, expected_module->body()[i]);
		i++;
	}

	spdlog::set_level(spdlog::level::debug);
	p.module()->print_node("");
	expected_module->print_node("");
	spdlog::set_level(spdlog::level::info);
}

std::shared_ptr<Module> create_test_module()
{
	return std::make_shared<Module>("_parser_test_.py");
}
}// namespace

TEST(Parser, SimplePositiveIntegerAssignment)
{
	constexpr std::string_view program = "a = 2\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<Constant>(int64_t{ 2 }),
		""));

	assert_generates_ast(program, expected_ast);
}

// TEST(Parser, MultipleAssignments)
// {
// 	constexpr std::string_view program = "a = b = 1\n";

// 	auto expected_ast = create_test_module();
// 	expected_ast->emplace(std::make_shared<Assign>(
// 		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Tuple>(
// 			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE),
// 				std::make_shared<Name>("b", ContextType::STORE) },
// 			ContextType::STORE) },
// 		std::make_shared<Constant>(int64_t{ 0 }),
// 		""));

// 	assert_generates_ast(program, expected_ast);
// }

TEST(Parser, MultipleAssignmentsWithStructuredBinding)
{
	constexpr std::string_view program = "a, b = 0, 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Tuple>(
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE),
				std::make_shared<Name>("b", ContextType::STORE) },
			ContextType::STORE) },
		std::make_shared<Tuple>(
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>(int64_t{ 0 }),
				std::make_shared<Constant>(int64_t{ 1 }) },
			ContextType::LOAD),
		""));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, BlankLine)
{
	constexpr std::string_view program =
		"a = 2\n"
		"\n"
		"b = 2\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<Constant>(static_cast<int64_t>(2)),
		""));
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("b", ContextType::STORE) },
		std::make_shared<Constant>(static_cast<int64_t>(2)),
		""));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SimplePositiveDoubleAssignment)
{
	constexpr std::string_view program = "a = 2.0\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<Constant>(static_cast<double>(2.0)),
		""));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SimpleStringAssignment)
{
	constexpr std::string_view program = "a = \"2\"\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<Constant>("2"),
		""));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, BinaryOperationWithAssignment)
{
	constexpr std::string_view program = "a = 1 + 2\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<BinaryExpr>(BinaryOpType::PLUS,
			std::make_shared<Constant>(static_cast<int64_t>(1)),
			std::make_shared<Constant>(static_cast<int64_t>(2))),
		""));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, BinaryOperationModulo)
{
	constexpr std::string_view program = "a = 3 % 4\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<BinaryExpr>(BinaryOpType::MODULO,
			std::make_shared<Constant>(int64_t{ 3 }),
			std::make_shared<Constant>(int64_t{ 4 })),
		""));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, FunctionDefinition)
{
	constexpr std::string_view program =
		"def add(a, b):\n"
		"   return a + b\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("add",// function_name
		std::make_shared<Arguments>(std::vector<std::shared_ptr<Argument>>{
			std::make_shared<Argument>("a", nullptr, ""),
			std::make_shared<Argument>("b", nullptr, ""),
		}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Return>(std::make_shared<BinaryExpr>(BinaryOpType::PLUS,
				std::make_shared<Name>("a", ContextType::LOAD),
				std::make_shared<Name>("b", ContextType::LOAD))),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		""// type_comment
		));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, FunctionDefinitionTypeAnnotation)
{
	constexpr std::string_view program =
		"def add(a: int, b: int) -> int:\n"
		"   return a + b\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("add",// function_name
		std::make_shared<Arguments>(std::vector<std::shared_ptr<Argument>>{
			std::make_shared<Argument>("a", std::make_shared<Name>("int", ContextType::LOAD), ""),
			std::make_shared<Argument>("b", std::make_shared<Name>("int", ContextType::LOAD), ""),
		}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Return>(std::make_shared<BinaryExpr>(BinaryOpType::PLUS,
				std::make_shared<Name>("a", ContextType::LOAD),
				std::make_shared<Name>("b", ContextType::LOAD))),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		std::make_shared<Name>("int", ContextType::LOAD),// returns
		""// type_comment
		));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, MultilineFunctionDefinition)
{
	constexpr std::string_view program =
		"def plus_one(a):\n"
		"   constant = 1\n"
		"   return a + constant\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("plus_one",// function_name
		std::make_shared<Arguments>(std::vector<std::shared_ptr<Argument>>{
			std::make_shared<Argument>("a", nullptr, ""),
		}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
										 "constant", ContextType::STORE) },
				std::make_shared<Constant>(static_cast<int64_t>(1)),
				""),
			std::make_shared<Return>(std::make_shared<BinaryExpr>(BinaryOpType::PLUS,
				std::make_shared<Name>("a", ContextType::LOAD),
				std::make_shared<Name>("constant", ContextType::LOAD))),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		""// type_comment
		));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, SimpleIfStatement)
{
	constexpr std::string_view program =
		"if True:\n"
		"   print(\"Hello, World!\")\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<If>(std::make_shared<Constant>(true),// test
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("print", ContextType::LOAD),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>("Hello, World!") },
			std::vector<std::shared_ptr<Keyword>>{}) },// body
		std::vector<std::shared_ptr<ASTNode>>{}// orelse
		));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SimpleIfElseStatement)
{
	constexpr std::string_view program =
		"if True:\n"
		"   print(\"Hello, World!\")\n"
		"else:\n"
		"   print(\"Goodbye!\")\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<If>(std::make_shared<Constant>(true),// test
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("print", ContextType::LOAD),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>("Hello, World!") },
			std::vector<std::shared_ptr<Keyword>>{}) },// body
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD),
				std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>("Goodbye!") },
				std::vector<std::shared_ptr<Keyword>>{}) }// orelse
		));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, IfStatementWithComparisson)
{
	constexpr std::string_view program =
		"a = 1\n"
		"if a == 1:\n"
		"   a = 2\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<Constant>(static_cast<int64_t>(1)),
		""));
	expected_ast->emplace(std::make_shared<If>(
		std::make_shared<Compare>(std::make_shared<Name>("a", ContextType::LOAD),
			Compare::OpType::Eq,
			std::make_shared<Constant>(static_cast<int64_t>(1))),// test
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
										 "a", ContextType::STORE) },
				std::make_shared<Constant>(int64_t{ 2 }),
				"") },// body
		std::vector<std::shared_ptr<ASTNode>>{}// orelse
		));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, LiteralList)
{
	constexpr std::string_view program = "a = [1, 2, 3, 5]\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<List>(
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 1 }),
				std::make_shared<Constant>(int64_t{ 2 }),
				std::make_shared<Constant>(int64_t{ 3 }),
				std::make_shared<Constant>(int64_t{ 5 }),
			},
			ContextType::LOAD),
		""));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, LiteralTuple)
{
	constexpr std::string_view program = "a = (1, 2, 3, 5)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<Tuple>(
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 1 }),
				std::make_shared<Constant>(int64_t{ 2 }),
				std::make_shared<Constant>(int64_t{ 3 }),
				std::make_shared<Constant>(int64_t{ 5 }),
			},
			ContextType::LOAD),
		""));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, LiteralDict)
{
	constexpr std::string_view program = "a = {\"a\": 1, b:2}\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<Dict>(
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>("a"),
				std::make_shared<Name>("b", ContextType::LOAD),
			},
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 1 }),
				std::make_shared<Constant>(int64_t{ 2 }),
			}),
		""));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SimpleForLoopWithFunctionCall)
{
	constexpr std::string_view program =
		"for x in range(10):\n"
		"	print(x)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<For>(
		std::make_shared<Name>("x", ContextType::STORE),// target
		std::make_shared<Call>(std::make_shared<Name>("range", ContextType::LOAD),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>(int64_t{ 10 }) },
			std::vector<std::shared_ptr<Keyword>>{}),// iter
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("print", ContextType::LOAD),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("x", ContextType::LOAD) },
			std::vector<std::shared_ptr<Keyword>>{}) },// body
		std::vector<std::shared_ptr<ASTNode>>{},// orelse
		""// type_comment
		));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ForLoopWithElseBlock)
{
	// you live and learn
	constexpr std::string_view program =
		"for x in range(10):\n"
		"	print(x)\n"
		"else:\n"
		"	print(\"ELSE!\")\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<For>(
		std::make_shared<Name>("x", ContextType::STORE),// target
		std::make_shared<Call>(std::make_shared<Name>("range", ContextType::LOAD),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>(int64_t{ 10 }) },
			std::vector<std::shared_ptr<Keyword>>{}),// iter
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("print", ContextType::LOAD),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("x", ContextType::LOAD) },
			std::vector<std::shared_ptr<Keyword>>{}) },// body
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD),
				std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>("ELSE!") },
				std::vector<std::shared_ptr<Keyword>>{}),
		},// orelse
		""// type_comment
		));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ClassDefinition)
{
	constexpr std::string_view program =
		"class A(Base, metaclass=MyMetaClass):\n"
		"	def __init__(self, value):\n"
		"		self.value = value\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<ClassDefinition>("A",// class name
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Name>("Base", ContextType::LOAD) },// bases
		std::vector{ std::make_shared<Keyword>(
			"metaclass", std::make_shared<Name>("MyMetaClass", ContextType::LOAD)) },// Keywords
		std::vector<std::shared_ptr<ast::ASTNode>>{
			std::make_shared<FunctionDefinition>("__init__",// function_name
				std::make_shared<Arguments>(std::vector<std::shared_ptr<Argument>>{
					std::make_shared<Argument>("self", nullptr, ""),
					std::make_shared<Argument>("value", nullptr, ""),
				}),// args
				std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Assign>(
					std::vector<std::shared_ptr<ast::ASTNode>>{ std::make_shared<Attribute>(
						std::make_shared<Name>("self", ContextType::LOAD),
						"value",
						ContextType::STORE) },
					std::make_shared<Name>("value", ContextType::LOAD),
					"") },// body
				std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
				nullptr,// returns
				""// type_comment
				) },// body
		std::vector<std::shared_ptr<ast::ASTNode>>{}// decorator_list
		));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AccessAttribute)
{
	constexpr std::string_view program = "test = foo.bar\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("test", ContextType::STORE) },
		std::make_shared<Attribute>(
			std::make_shared<Name>("foo", ContextType::LOAD), "bar", ContextType::LOAD),
		""));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, CallMethod)
{
	constexpr std::string_view program = "test = foo.bar()\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("test", ContextType::STORE) },
		std::make_shared<Call>(std::make_shared<Attribute>(
			std::make_shared<Name>("foo", ContextType::LOAD), "bar", ContextType::LOAD)),
		""));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, LiteralMethodCall)
{
	constexpr std::string_view program = "\"foo123\".isalnum()\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(std::make_shared<Attribute>(
		std::make_shared<Constant>("foo123"), "isalnum", ContextType::LOAD)));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionCallWithKwarg)
{
	constexpr std::string_view program = "print(\"Hello\", \"world!\", sep=',')\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD),
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Constant>("Hello"), std::make_shared<Constant>("world!") },
		std::vector{ std::make_shared<Keyword>("sep", std::make_shared<Constant>(",")) }));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionCallWithOnlyKwargs)
{
	constexpr std::string_view program = "add(a=1, b=2)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(std::make_shared<Name>("add", ContextType::LOAD),
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector{ std::make_shared<Keyword>("a", std::make_shared<Constant>(int64_t{ 1 })),
			std::make_shared<Keyword>("b", std::make_shared<Constant>(int64_t{ 2 })) }));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionCallWithKwargAsResultFromAnotherFunction)
{
	constexpr std::string_view program = "print(\"Hello\", \"world!\", sep=my_separator())\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD),
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Constant>("Hello"), std::make_shared<Constant>("world!") },
		std::vector{ std::make_shared<Keyword>("sep",
			std::make_shared<Call>(std::make_shared<Name>("my_separator", ContextType::LOAD))) }));
	assert_generates_ast(program, expected_ast);
}


TEST(Parser, AugmentedAssign)
{
	constexpr std::string_view program = "a += b\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<AugAssign>(std::make_shared<Name>("a", ContextType::STORE),
			BinaryOpType::PLUS,
			std::make_shared<Name>("b", ContextType::LOAD)));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, WhileLoop)
{
	constexpr std::string_view program =
		"while a <= 10:\n"
		"  a += 1\n"
		"else:\n"
		"  print(a)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<While>(
		std::make_shared<Compare>(std::make_shared<Name>("a", ContextType::LOAD),
			Compare::OpType::LtE,
			std::make_shared<Constant>(int64_t{ 10 })),
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<AugAssign>(std::make_shared<Name>("a", ContextType::STORE),
				BinaryOpType::PLUS,
				std::make_shared<Constant>(int64_t{ 1 })) },
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("print", ContextType::LOAD),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::LOAD) },
			std::vector<std::shared_ptr<Keyword>>{}) }));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, Import)
{
	constexpr std::string_view program = "import fibo\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Import>(std::vector<std::string>{ "fibo" }));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportAs)
{
	constexpr std::string_view program = "import fibo as f\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Import>(std::vector<std::string>{ "fibo" }, "f"));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportDottedAs)
{
	constexpr std::string_view program = "import fibo.nac.ci as f\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Import>(std::vector<std::string>{ "fibo", "nac", "ci" }, "f"));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SubscriptIndexExpression)
{
	constexpr std::string_view program = "a[0]\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Subscript>(std::make_shared<Name>("a", ContextType::LOAD),
			Subscript::Index{ std::make_shared<Constant>(int64_t{ 0 }) },
			ContextType::LOAD));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, SubscriptIndexAssignment)
{
	constexpr std::string_view program = "a[0] = 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Subscript>(
									 std::make_shared<Name>("a", ContextType::LOAD),
									 Subscript::Index{ std::make_shared<Constant>(int64_t{ 0 }) },
									 ContextType::STORE) },
			std::make_shared<Constant>(int64_t{ 1 }),
			""));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, SubscriptSliceExpression)
{
	constexpr std::string_view program =
		"a[0:10]\n"
		"b[0:10:2]\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Subscript>(std::make_shared<Name>("a", ContextType::LOAD),
			Subscript::Slice{ .lower = std::make_shared<Constant>(int64_t{ 0 }),
				.upper = std::make_shared<Constant>(int64_t{ 10 }) },
			ContextType::LOAD));
	expected_ast->emplace(
		std::make_shared<Subscript>(std::make_shared<Name>("b", ContextType::LOAD),
			Subscript::Slice{ .lower = std::make_shared<Constant>(int64_t{ 0 }),
				.upper = std::make_shared<Constant>(int64_t{ 10 }),
				.step = std::make_shared<Constant>(int64_t{ 2 }) },
			ContextType::LOAD));


	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SubscriptSliceAssignment)
{
	constexpr std::string_view program =
		"a[0:10] = c\n"
		"b[0:10:2] = d\n"
		"b[0:g:2] = f[e:h]\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Subscript>(std::make_shared<Name>("a", ContextType::LOAD),
				Subscript::Slice{ .lower = std::make_shared<Constant>(int64_t{ 0 }),
					.upper = std::make_shared<Constant>(int64_t{ 10 }) },
				ContextType::STORE) },
		std::make_shared<Name>("c", ContextType::LOAD),
		""));
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Subscript>(std::make_shared<Name>("b", ContextType::LOAD),
				Subscript::Slice{ .lower = std::make_shared<Constant>(int64_t{ 0 }),
					.upper = std::make_shared<Constant>(int64_t{ 10 }),
					.step = std::make_shared<Constant>(int64_t{ 2 }) },
				ContextType::STORE) },
		std::make_shared<Name>("d", ContextType::LOAD),
		""));
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Subscript>(std::make_shared<Name>("b", ContextType::LOAD),
				Subscript::Slice{ .lower = std::make_shared<Constant>(int64_t{ 0 }),
					.upper = std::make_shared<Name>("g", ContextType::LOAD),
					.step = std::make_shared<Constant>(int64_t{ 2 }) },
				ContextType::STORE) },
		std::make_shared<Subscript>(std::make_shared<Name>("f", ContextType::LOAD),
			Subscript::Slice{ .lower = std::make_shared<Name>("e", ContextType::LOAD),
				.upper = std::make_shared<Name>("h", ContextType::LOAD) },
			ContextType::LOAD),
		""));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, Raise)
{
	constexpr std::string_view program = "raise\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Raise>());

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, RaiseValueError)
{
	constexpr std::string_view program = "raise ValueError(\"Wrong!\")\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Raise>(
		std::make_shared<Call>(std::make_shared<Name>("ValueError", ContextType::LOAD),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>("Wrong!") },
			std::vector<std::shared_ptr<Keyword>>{}),
		nullptr));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, RaiseValueErrorCause)
{
	constexpr std::string_view program = "raise ValueError(\"Wrong!\") from exc\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Raise>(
		std::make_shared<Call>(std::make_shared<Name>("ValueError", ContextType::LOAD),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>("Wrong!") },
			std::vector<std::shared_ptr<Keyword>>{}),
		std::make_shared<Name>("exc", ContextType::LOAD)));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, TryFinally)
{
	constexpr std::string_view program =
		"try:\n"
		"  foo()\n"
		"finally:\n"
		"  bar()\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Try>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
								  std::make_shared<Name>("foo", ContextType::LOAD)) },
			std::vector<std::shared_ptr<ExceptHandler>>{},
			std::vector<std::shared_ptr<ASTNode>>{},
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Call>(std::make_shared<Name>("bar", ContextType::LOAD)) }));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, TryExcept)
{
	constexpr std::string_view program =
		"try:\n"
		"  foo()\n"
		"except:\n"
		"  print(\"Exception\")\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Try>(
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD)) },
		std::vector{ std::make_shared<ExceptHandler>(nullptr,
			"",
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
				std::make_shared<Name>("print", ContextType::LOAD),
				std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>("Exception") },
				std::vector<std::shared_ptr<Keyword>>{}) }) },
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<ASTNode>>{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, TryExceptWithExceptionType)
{
	constexpr std::string_view program =
		"try:\n"
		"  foo()\n"
		"except ValueError:\n"
		"  print(\"Exception\")\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Try>(
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD)) },
		std::vector{ std::make_shared<ExceptHandler>(
			std::make_shared<Name>("ValueError", ContextType::LOAD),
			"",
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
				std::make_shared<Name>("print", ContextType::LOAD),
				std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>("Exception") },
				std::vector<std::shared_ptr<Keyword>>{}) }) },
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<ASTNode>>{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, TryExceptWithExceptionTypeAndName)
{
	constexpr std::string_view program =
		"try:\n"
		"  foo()\n"
		"except ValueError as e:\n"
		"  print(e)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Try>(
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD)) },
		std::vector{
			std::make_shared<ExceptHandler>(std::make_shared<Name>("ValueError", ContextType::LOAD),
				"e",
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD),
						std::vector<std::shared_ptr<ASTNode>>{
							std::make_shared<Name>("e", ContextType::LOAD) },
						std::vector<std::shared_ptr<Keyword>>{}) }) },
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<ASTNode>>{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, TryExceptMultipleExceptionHandlers)
{
	constexpr std::string_view program =
		"try:\n"
		"  foo()\n"
		"except ValueError as e:\n"
		"  print(e)\n"
		"except BaseException:\n"
		"  print(\"BaseException\")\n"
		"except:\n"
		"  print(\"Exception\")\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Try>(
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD)) },
		std::vector{
			std::make_shared<ExceptHandler>(std::make_shared<Name>("ValueError", ContextType::LOAD),
				"e",
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD),
						std::vector<std::shared_ptr<ASTNode>>{
							std::make_shared<Name>("e", ContextType::LOAD) },
						std::vector<std::shared_ptr<Keyword>>{}) }),
			std::make_shared<ExceptHandler>(
				std::make_shared<Name>("BaseException", ContextType::LOAD),
				"",
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD),
						std::vector<std::shared_ptr<ASTNode>>{
							std::make_shared<Constant>("BaseException") },
						std::vector<std::shared_ptr<Keyword>>{}) }),
			std::make_shared<ExceptHandler>(nullptr,
				"",
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD),
						std::vector<std::shared_ptr<ASTNode>>{
							std::make_shared<Constant>("Exception") },
						std::vector<std::shared_ptr<Keyword>>{}) }) },
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<ASTNode>>{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, TryExceptFinally)
{
	constexpr std::string_view program =
		"try:\n"
		"  foo()\n"
		"except ValueError as e:\n"
		"  print(e)\n"
		"finally:\n"
		"  cleanup()\n"
		"  exit()\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Try>(
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD)) },
		std::vector{
			std::make_shared<ExceptHandler>(std::make_shared<Name>("ValueError", ContextType::LOAD),
				"e",
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD),
						std::vector<std::shared_ptr<ASTNode>>{
							std::make_shared<Name>("e", ContextType::LOAD) },
						std::vector<std::shared_ptr<Keyword>>{}) }) },
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Call>(std::make_shared<Name>("cleanup", ContextType::LOAD)),
			std::make_shared<Call>(std::make_shared<Name>("exit", ContextType::LOAD)) }));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, Assert)
{
	constexpr std::string_view program = "assert False\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assert>(std::make_shared<Constant>(false), nullptr));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AssertWithMessage)
{
	constexpr std::string_view program = "assert False, \"failed!\"\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assert>(
		std::make_shared<Constant>(false), std::make_shared<Constant>("failed!")));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, NegativeNumber)
{
	constexpr std::string_view program = "a = -1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<UnaryExpr>(UnaryOpType::SUB, std::make_shared<Constant>(int64_t{ 1 })),
		""));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, UnaryExprMix)
{
	constexpr std::string_view program = "a = -+-+1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<UnaryExpr>(UnaryOpType::SUB,
			std::make_shared<UnaryExpr>(UnaryOpType::ADD,
				std::make_shared<UnaryExpr>(UnaryOpType::SUB,
					std::make_shared<UnaryExpr>(
						UnaryOpType::ADD, std::make_shared<Constant>(int64_t{ 1 }))))),
		""));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, UnaryNot)
{
	constexpr std::string_view program = "a = not b\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<UnaryExpr>(
			UnaryOpType::NOT, std::make_shared<Name>("b", ContextType::LOAD)),
		""));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, CompareIsNot)
{
	constexpr std::string_view program = "assert a is not False\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assert>(
		std::make_shared<Compare>(std::make_shared<Name>("a", ContextType::LOAD),
			Compare::OpType::IsNot,
			std::make_shared<Constant>(false)),
		nullptr));
	assert_generates_ast(program, expected_ast);
}


TEST(Parser, CompareIs)
{
	constexpr std::string_view program = "assert a is False\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assert>(
		std::make_shared<Compare>(std::make_shared<Name>("a", ContextType::LOAD),
			Compare::OpType::Is,
			std::make_shared<Constant>(false)),
		nullptr));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, BoolOpAnd)
{
	constexpr std::string_view program = "a and b\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<BoolOp>(BoolOp::OpType::And,
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::LOAD),
			std::make_shared<Name>("b", ContextType::LOAD) }));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, BoolOpOr)
{
	constexpr std::string_view program = "a or b\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<BoolOp>(BoolOp::OpType::Or,
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::LOAD),
			std::make_shared<Name>("b", ContextType::LOAD) }));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, WithStatement)
{
	constexpr std::string_view program =
		"with lock:\n"
		"  work()\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<With>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("lock", ContextType::LOAD) },
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Call>(std::make_shared<Name>("work", ContextType::LOAD)) },
		""));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, IfExpression)
{
	constexpr std::string_view program = "a = 42 if magic() else 21\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("a", ContextType::STORE) },
		std::make_shared<IfExpr>(
			std::make_shared<Call>(std::make_shared<Name>("magic", ContextType::LOAD)),
			std::make_shared<Constant>(int64_t{ 42 }),
			std::make_shared<Constant>(int64_t{ 21 })),
		""));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ArgsKwargsFunctionDef)
{
	constexpr std::string_view program =
		"def foo(*args, **kwargs):\n"
		"  return 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("foo",
		std::make_shared<Arguments>(std::vector<std::shared_ptr<Argument>>{},
			std::vector<std::shared_ptr<Argument>>{},
			std::make_shared<Argument>("args", nullptr, ""),
			std::vector<std::shared_ptr<Argument>>{},
			std::vector<std::shared_ptr<ast::ASTNode>>{},
			std::make_shared<Argument>("kwargs", nullptr, ""),
			std::vector<std::shared_ptr<ast::ASTNode>>{}),
		std::vector<std::shared_ptr<ast::ASTNode>>{
			std::make_shared<Return>(std::make_shared<Constant>(int64_t{ 1 })) },
		std::vector<std::shared_ptr<ast::ASTNode>>{},
		nullptr,
		""));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ArgsKwargsFunctionCall)
{
	constexpr std::string_view program = "foo(*args, **kwargs)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD),
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Starred>(
			std::make_shared<Name>("args", ContextType::LOAD), ContextType::LOAD) },
		std::vector<std::shared_ptr<Keyword>>{
			std::make_shared<Keyword>(std::make_shared<Name>("kwargs", ContextType::LOAD)) }));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ArgKwargsFunctionCall)
{
	constexpr std::string_view program = "foo(arg, **kwargs)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD),
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>("arg", ContextType::LOAD) },
		std::vector<std::shared_ptr<Keyword>>{
			std::make_shared<Keyword>(std::make_shared<Name>("kwargs", ContextType::LOAD)) }));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ArgsKwargsArgFunctionCall)
{
	constexpr std::string_view program = "foo(*args, **kwargs, a=1)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD),
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Starred>(
			std::make_shared<Name>("args", ContextType::LOAD), ContextType::LOAD) },
		std::vector<std::shared_ptr<Keyword>>{
			std::make_shared<Keyword>(std::make_shared<Name>("kwargs", ContextType::LOAD)),
			std::make_shared<Keyword>("a", std::make_shared<Constant>(int64_t{ 1 })) }));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ComplexArgsKwargsFunctionCall)
{
	constexpr std::string_view program =
		"foo(*args, *more_args, bar, b=2, **kwargs, **more_kwargs, a=1)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD),
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Starred>(
				std::make_shared<Name>("args", ContextType::LOAD), ContextType::LOAD),
			std::make_shared<Starred>(
				std::make_shared<Name>("more_args", ContextType::LOAD), ContextType::LOAD),
			std::make_shared<Name>("bar", ContextType::LOAD) },
		std::vector<std::shared_ptr<Keyword>>{
			std::make_shared<Keyword>("b", std::make_shared<Constant>(int64_t{ 2 })),
			std::make_shared<Keyword>(std::make_shared<Name>("kwargs", ContextType::LOAD)),
			std::make_shared<Keyword>(std::make_shared<Name>("more_kwargs", ContextType::LOAD)),
			std::make_shared<Keyword>("a", std::make_shared<Constant>(int64_t{ 1 })) }));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AugAssignAttribute)
{
	constexpr std::string_view program = "a.b += 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<AugAssign>(
		std::make_shared<Attribute>(
			std::make_shared<Name>("a", ContextType::LOAD), "b", ContextType::STORE),
		BinaryOpType::PLUS,
		std::make_shared<Constant>(int64_t{ 1 })));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AugAssignSlices)
{
	constexpr std::string_view program = "a['b'] += 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<AugAssign>(
		std::make_shared<Subscript>(std::make_shared<Name>("a", ContextType::LOAD),
			Subscript::Index{ .value = std::make_shared<Constant>("b") },
			ContextType::STORE),
		BinaryOpType::PLUS,
		std::make_shared<Constant>(int64_t{ 1 })));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionDefinitionWithDecoratorList)
{
	constexpr std::string_view program =
		"@classmethod\n"
		"@_require_frozen\n"
		"def get_source(cls):\n"
		"  return None\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("get_source",// function_name
		std::make_shared<Arguments>(std::vector<std::shared_ptr<Argument>>{
			std::make_shared<Argument>("cls", nullptr, ""),
		}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Return>(std::make_shared<Constant>(NoneType{})),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Name>("classmethod", ContextType::LOAD),
			std::make_shared<Name>("_require_frozen", ContextType::LOAD),
		},// decorator_list
		nullptr,// returns
		""// type_comment
		));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ClassDefinitionWithDecoratorList)
{
	constexpr std::string_view program =
		"@my_decorator\n"
		"class Derived(Base):\n"
		"  pass\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<ClassDefinition>("Derived",// class_name
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Name>("Base", ContextType::LOAD),
		},// bases
		std::vector<std::shared_ptr<Keyword>>{},// keyword
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Pass>() },// body
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Name>("my_decorator", ContextType::LOAD),
		}// decorator_list
		));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, NamedExpression)
{
	constexpr std::string_view program =
		"if a:=1:\n"
		"  pass\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<If>(
		std::make_shared<NamedExpr>(std::make_shared<Name>("a", ContextType::STORE),
			std::make_shared<Constant>(int64_t{ 1 })),
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Pass>() },
		std::vector<std::shared_ptr<ASTNode>>{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionDefinitionWithDefaultKeyword)
{
	constexpr std::string_view program =
		"def f(a, b=1):\n"
		"  pass\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("f",// function_name
		std::make_shared<Arguments>(std::vector<std::shared_ptr<ast::Argument>>{},
			std::vector{
				std::make_shared<Argument>("a", nullptr, ""),
				std::make_shared<Argument>("b", nullptr, ""),
			},
			nullptr,
			std::vector<std::shared_ptr<ast::Argument>>{},
			std::vector<std::shared_ptr<ASTNode>>{},
			nullptr,
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 1 }) }),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Pass>(),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		""// type_comment
		));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionDefinitionWithKeywordOnlyArg)
{
	constexpr std::string_view program =
		"def f(a, *, b=1):\n"
		"  pass\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("f",// function_name
		std::make_shared<Arguments>(std::vector<std::shared_ptr<ast::Argument>>{},
			std::vector{
				std::make_shared<Argument>("a", nullptr, ""),
			},
			nullptr,
			std::vector{ std::make_shared<ast::Argument>("b", nullptr, "") },
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>(int64_t{ 1 }) },
			nullptr,
			std::vector<std::shared_ptr<ASTNode>>{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Pass>(),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		""// type_comment
		));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionDefinitionWithDefaultArgsAndKeywordOnlyArg)
{
	constexpr std::string_view program =
		"def f(a, b=1, *, c=2, **kwargs):\n"
		"  pass\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("f",// function_name
		std::make_shared<Arguments>(std::vector<std::shared_ptr<ast::Argument>>{},
			std::vector{
				std::make_shared<Argument>("a", nullptr, ""),
				std::make_shared<Argument>("b", nullptr, ""),
			},
			nullptr,
			std::vector{ std::make_shared<ast::Argument>("c", nullptr, "") },
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>(int64_t{ 2 }) },
			std::make_shared<Argument>("kwargs", nullptr, ""),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 1 }) }),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Pass>(),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		""// type_comment
		));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionDefinitionWithDefaultArgsAndKeywordOnlyArgNone)
{
	constexpr std::string_view program =
		"def f(a, b=1, *, c, d, **kwargs):\n"
		"  pass\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("f",// function_name
		std::make_shared<Arguments>(std::vector<std::shared_ptr<ast::Argument>>{},
			std::vector{
				std::make_shared<Argument>("a", nullptr, ""),
				std::make_shared<Argument>("b", nullptr, ""),
			},
			nullptr,
			std::vector{ std::make_shared<ast::Argument>("c", nullptr, ""),
				std::make_shared<ast::Argument>("d", nullptr, "") },
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(NoneType{}), std::make_shared<Constant>(NoneType{}) },
			std::make_shared<Argument>("kwargs", nullptr, ""),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 1 }) }),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Pass>(),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		""// type_comment
		));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AssignToAttributeSubscript)
{
	constexpr std::string_view program = "sys.modules.foo[spec.name.bar] = my.module\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Subscript>(
			std::make_shared<Attribute>(
				std::make_shared<Attribute>(
					std::make_shared<Name>("sys", ContextType::LOAD), "modules", ContextType::LOAD),
				"foo",
				ContextType::LOAD),
			Subscript::Index{
				.value = std::make_shared<Attribute>(
					std::make_shared<Attribute>(std::make_shared<Name>("spec", ContextType::LOAD),
						"name",
						ContextType::LOAD),
					"bar",
					ContextType::LOAD) },
			ContextType::STORE) },
		std::make_shared<Attribute>(
			std::make_shared<Name>("my", ContextType::LOAD), "module", ContextType::LOAD),
		""));
	assert_generates_ast(program, expected_ast);
}
