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
					   if (auto *int_result = std::get_if<BigIntType>(&number_value.value)) {
						   ASSERT_EQ(*int_result,
							   std::get<BigIntType>(std::get<Number>(*expected_value).value));
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
				} else if (std::holds_alternative<NoneType>(
							   std::get<NameConstant>(*expected_value).value)) {
					ASSERT_TRUE(std::holds_alternative<NoneType>(name_constant_value.value));
				} else {
					TODO();
				}
			},
			[&](const Bytes &bytes) { ASSERT_EQ(bytes.b, std::get<Bytes>(*expected_value).b); },
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

void compare_async_function_definition(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::AsyncFunctionDefinition);

	const auto result_name = as<AsyncFunctionDefinition>(result)->name();
	const auto expected_name = as<AsyncFunctionDefinition>(expected)->name();
	EXPECT_EQ(result_name, expected_name);

	const auto result_args = as<AsyncFunctionDefinition>(result)->args();
	const auto expected_args = as<AsyncFunctionDefinition>(expected)->args();
	dispatch(result_args, expected_args);

	const auto result_body = as<AsyncFunctionDefinition>(result)->body();
	const auto expected_body = as<AsyncFunctionDefinition>(expected)->body();
	ASSERT_EQ(result_body.size(), expected_body.size());
	for (size_t i = 0; i < result_body.size(); ++i) { dispatch(result_body[i], expected_body[i]); }

	const auto result_decorator_list = as<AsyncFunctionDefinition>(result)->decorator_list();
	const auto expected_decorator_list = as<AsyncFunctionDefinition>(expected)->decorator_list();
	ASSERT_EQ(result_decorator_list.size(), expected_decorator_list.size());
	for (size_t i = 0; i < result_decorator_list.size(); ++i) {
		dispatch(result_decorator_list[i], expected_decorator_list[i]);
	}

	const auto result_returns = as<AsyncFunctionDefinition>(result)->returns();
	const auto expected_returns = as<AsyncFunctionDefinition>(expected)->returns();
	dispatch(result_returns, expected_returns);

	const auto result_type_comment = as<AsyncFunctionDefinition>(result)->type_comment();
	const auto expected_type_comment = as<AsyncFunctionDefinition>(expected)->type_comment();
	EXPECT_EQ(result_type_comment, expected_type_comment);
}

void compare_lambda(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Lambda);

	const auto result_args = as<Lambda>(result)->args();
	const auto expected_args = as<Lambda>(expected)->args();
	dispatch(result_args, expected_args);

	const auto result_body = as<Lambda>(result)->body();
	const auto expected_body = as<Lambda>(expected)->body();
	dispatch(result_body, expected_body);
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

void compare_yield(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Yield);

	const auto result_value = as<Yield>(result)->value();
	const auto expected_value = as<Yield>(expected)->value();
	dispatch(result_value, expected_value);
}

void compare_yieldfrom(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::YieldFrom);

	const auto result_value = as<YieldFrom>(result)->value();
	const auto expected_value = as<YieldFrom>(expected)->value();
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

	const auto result_context = as<ast::Tuple>(result)->context();
	const auto expected_context = as<ast::Tuple>(expected)->context();
	ASSERT_EQ(result_context, expected_context);

	const auto result_elements = as<ast::Tuple>(result)->elements();
	const auto expected_elements = as<ast::Tuple>(expected)->elements();
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
		ASSERT_EQ(result_names[i].name, expected_names[i].name);
		ASSERT_EQ(result_names[i].asname, expected_names[i].asname);
	}
}

void compare_import_from(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::ImportFrom);

	const auto result_names = as<ImportFrom>(result)->names();
	const auto expected_names = as<ImportFrom>(expected)->names();
	ASSERT_EQ(result_names.size(), expected_names.size());

	for (size_t i = 0; i < result_names.size(); ++i) {
		ASSERT_EQ(result_names[i].name, expected_names[i].name);
		ASSERT_EQ(result_names[i].asname, expected_names[i].asname);
	}

	ASSERT_EQ(as<ImportFrom>(result)->level(), as<ImportFrom>(expected)->level());
	ASSERT_EQ(as<ImportFrom>(result)->module(), as<ImportFrom>(expected)->module());
}

void compare_slices(const Subscript::SliceType &result, const Subscript::SliceType &expected)
{
	ASSERT_EQ(result.index(), expected.index());

	if (std::holds_alternative<Subscript::Slice>(result)) {
		auto result_s = std::get<Subscript::Slice>(result);
		auto expected_s = std::get<Subscript::Slice>(expected);

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
	} else if (std::holds_alternative<Subscript::Index>(expected)) {
		auto result_i = std::get<Subscript::Index>(result);
		auto expected_i = std::get<Subscript::Index>(expected);

		if (expected_i.value) {
			ASSERT_TRUE(result_i.value);
			dispatch(result_i.value, expected_i.value);
		} else {
			ASSERT_FALSE(result_i.value);
		}

	} else {
		auto result_e = std::get<Subscript::ExtSlice>(result);
		auto expected_e = std::get<Subscript::ExtSlice>(expected);

		ASSERT_EQ(result_e.dims.size(), expected_e.dims.size());
		for (size_t i = 0; i < result_e.dims.size(); ++i) {
			std::visit([](const auto &lhs, const auto &rhs) { compare_slices(lhs, rhs); },
				result_e.dims[i],
				expected_e.dims[i]);
		}
	}
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

	compare_slices(result_slice, expected_slice);

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

void compare_pass(const std::shared_ptr<ASTNode> &result, const std::shared_ptr<ASTNode> &)
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

void compare_with_item(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::WithItem);

	const auto result_context_expr = as<WithItem>(result)->context_expr();
	const auto expected_context_expr = as<WithItem>(expected)->context_expr();
	dispatch(result_context_expr, expected_context_expr);

	const auto result_optional_vars = as<WithItem>(result)->optional_vars();
	const auto expected_optional_vars = as<WithItem>(expected)->optional_vars();
	dispatch(result_optional_vars, expected_optional_vars);
}

void compare_list_comprehension(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::ListComp);

	const auto result_elt = as<ListComp>(result)->elt();
	const auto expected_elt = as<ListComp>(expected)->elt();
	dispatch(result_elt, expected_elt);

	const auto result_generators = as<ListComp>(result)->generators();
	const auto expected_generators = as<ListComp>(expected)->generators();
	ASSERT_EQ(result_generators.size(), expected_generators.size());
	for (size_t i = 0; i < result_generators.size(); ++i) {
		dispatch(result_generators[i], expected_generators[i]);
	}
}

void compare_dict_comprehension(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::DictComp);

	const auto result_key = as<DictComp>(result)->key();
	const auto expected_key = as<DictComp>(expected)->key();
	dispatch(result_key, expected_key);

	const auto result_value = as<DictComp>(result)->value();
	const auto expected_value = as<DictComp>(expected)->value();
	dispatch(result_value, expected_value);

	const auto result_generators = as<DictComp>(result)->generators();
	const auto expected_generators = as<DictComp>(expected)->generators();
	ASSERT_EQ(result_generators.size(), expected_generators.size());
	for (size_t i = 0; i < result_generators.size(); ++i) {
		dispatch(result_generators[i], expected_generators[i]);
	}
}

void compare_set_comprehension(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::SetComp);

	const auto result_elt = as<SetComp>(result)->elt();
	const auto expected_elt = as<SetComp>(expected)->elt();
	dispatch(result_elt, expected_elt);

	const auto result_generators = as<SetComp>(result)->generators();
	const auto expected_generators = as<SetComp>(expected)->generators();
	ASSERT_EQ(result_generators.size(), expected_generators.size());
	for (size_t i = 0; i < result_generators.size(); ++i) {
		dispatch(result_generators[i], expected_generators[i]);
	}
}

void compare_comprehension(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::Comprehension);

	const auto result_target = as<Comprehension>(result)->target();
	const auto expected_target = as<Comprehension>(expected)->target();
	dispatch(result_target, expected_target);

	const auto result_iter = as<Comprehension>(result)->iter();
	const auto expected_iter = as<Comprehension>(expected)->iter();
	dispatch(result_iter, expected_iter);

	const auto result_ifs = as<Comprehension>(result)->ifs();
	const auto expected_ifs = as<Comprehension>(expected)->ifs();
	ASSERT_EQ(result_ifs.size(), expected_ifs.size());
	for (size_t i = 0; i < result_ifs.size(); ++i) { dispatch(result_ifs[i], expected_ifs[i]); }

	ASSERT_EQ(as<Comprehension>(result)->is_async(), as<Comprehension>(expected)->is_async());
}

void compare_generator_expression(const std::shared_ptr<ASTNode> &result,
	const std::shared_ptr<ASTNode> &expected)
{
	ASSERT_EQ(result->node_type(), ASTNodeType::GeneratorExp);

	const auto result_elt = as<GeneratorExp>(result)->elt();
	const auto expected_elt = as<GeneratorExp>(expected)->elt();
	dispatch(result_elt, expected_elt);

	const auto result_generators = as<GeneratorExp>(result)->generators();
	const auto expected_generators = as<GeneratorExp>(expected)->generators();
	ASSERT_EQ(result_generators.size(), expected_generators.size());
	for (size_t i = 0; i < result_generators.size(); ++i) {
		dispatch(result_generators[i], expected_generators[i]);
	}
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
	case ASTNodeType::AsyncFunctionDefinition: {
		compare_async_function_definition(result, expected);
		break;
	}
	case ASTNodeType::Return: {
		compare_return(result, expected);
		break;
	}
	case ASTNodeType::Yield: {
		compare_yield(result, expected);
		break;
	}
	case ASTNodeType::YieldFrom: {
		compare_yieldfrom(result, expected);
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
	case ASTNodeType::ImportFrom: {
		compare_import_from(result, expected);
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
	case ASTNodeType::WithItem: {
		compare_with_item(result, expected);
		break;
	}
	case ASTNodeType::ListComp: {
		compare_list_comprehension(result, expected);
		break;
	}
	case ASTNodeType::DictComp: {
		compare_dict_comprehension(result, expected);
		break;
	}
	case ASTNodeType::SetComp: {
		compare_set_comprehension(result, expected);
		break;
	}
	case ASTNodeType::Comprehension: {
		compare_comprehension(result, expected);
		break;
	}
	case ASTNodeType::GeneratorExp: {
		compare_generator_expression(result, expected);
		break;
	}
	case ASTNodeType::Lambda: {
		compare_lambda(result, expected);
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
	p.parse();
	ASSERT_TRUE(p.module());

	const auto lvl = spdlog::get_level();
	spdlog::set_level(spdlog::level::debug);
	p.module()->print_node("");
	expected_module->print_node("");
	spdlog::set_level(lvl);

	ASSERT_EQ(p.module()->body().size(), expected_module->body().size());

	size_t i = 0;
	for (const auto &node : p.module()->body()) {
		dispatch(node, expected_module->body()[i]);
		i++;
	}
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
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, MultipleAssignments)
{
	constexpr std::string_view program = "a = b = 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Name>("a", ContextType::STORE, SourceLocation{}),
			std::make_shared<Name>("b", ContextType::STORE, SourceLocation{}),
		},
		std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
		"",
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, MultipleAssignmentsWithStructuredBinding)
{
	constexpr std::string_view program = "a, b = 0, 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<ast::Tuple>(
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Name>("a", ContextType::STORE, SourceLocation{}),
				std::make_shared<Name>("b", ContextType::STORE, SourceLocation{}) },
			ContextType::STORE,
			SourceLocation{}) },
		std::make_shared<ast::Tuple>(
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 0 }, SourceLocation{}),
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}) },
			ContextType::LOAD,
			SourceLocation{}),
		"",
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SingleValueAssignmentTuple)
{
	constexpr std::string_view program = "_CASE_INSENSITIVE_PLATFORMS_STR_KEY = 'win',\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
			"_CASE_INSENSITIVE_PLATFORMS_STR_KEY", ContextType::STORE, SourceLocation{}) },
		std::make_shared<ast::Tuple>(
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(String{ "win" }, SourceLocation{}),
			},
			ContextType::LOAD,
			SourceLocation{}),
		"",
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AssignToTuple)
{
	constexpr std::string_view program = "(a, b) = foo\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<ast::Tuple>(
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Name>("a", ContextType::STORE, SourceLocation{}),
				std::make_shared<Name>("b", ContextType::STORE, SourceLocation{}),
			},
			ContextType::STORE,
			SourceLocation{}) },
		std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}),
		"",
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AssignToList)
{
	constexpr std::string_view program = "[a, b] = foo\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<List>(
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Name>("a", ContextType::STORE, SourceLocation{}),
				std::make_shared<Name>("b", ContextType::STORE, SourceLocation{}),
			},
			ContextType::STORE,
			SourceLocation{}) },
		std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}),
		"",
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, BlankLine)
{
	constexpr std::string_view program =
		"a = 2\n"
		"\n"
		"b = 2\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Constant>(static_cast<int64_t>(2), SourceLocation{}),
			"",
			SourceLocation{}));
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "b", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Constant>(static_cast<int64_t>(2), SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SimplePositiveDoubleAssignment)
{
	constexpr std::string_view program = "a = 2.0\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Constant>(2.0, SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SimpleStringAssignment)
{
	constexpr std::string_view program = "a = \"2\"\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Constant>("2", SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, BinaryOperationWithAssignment)
{
	constexpr std::string_view program = "a = 1 + 2\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<BinaryExpr>(BinaryOpType::PLUS,
				std::make_shared<Constant>(static_cast<int64_t>(1), SourceLocation{}),
				std::make_shared<Constant>(static_cast<int64_t>(2), SourceLocation{}),
				SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, BinaryOperationModulo)
{
	constexpr std::string_view program = "a = 3 % 4\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<BinaryExpr>(BinaryOpType::MODULO,
				std::make_shared<Constant>(int64_t{ 3 }, SourceLocation{}),
				std::make_shared<Constant>(int64_t{ 4 }, SourceLocation{}),
				SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, BinaryOperationAnd)
{
	constexpr std::string_view program = "a = 0 & 1\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<BinaryExpr>(BinaryOpType::AND,
				std::make_shared<Constant>(int64_t{ 0 }, SourceLocation{}),
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
				SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, BinaryOperationOr)
{
	constexpr std::string_view program = "a = 0 | 1\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<BinaryExpr>(BinaryOpType::OR,
				std::make_shared<Constant>(int64_t{ 0 }, SourceLocation{}),
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
				SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, BinaryOperationXor)
{
	constexpr std::string_view program = "a = 0 ^ 1\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<BinaryExpr>(BinaryOpType::XOR,
				std::make_shared<Constant>(int64_t{ 0 }, SourceLocation{}),
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
				SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionDefinition)
{
	constexpr std::string_view program =
		"def add(a, b):\n"
		"   return a + b\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("add",// function_name
		std::make_shared<Arguments>(
			std::vector<std::shared_ptr<Argument>>{
				std::make_shared<Argument>("a", nullptr, "", SourceLocation{}),
				std::make_shared<Argument>("b", nullptr, "", SourceLocation{}),
			},
			SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Return>(
				std::make_shared<BinaryExpr>(BinaryOpType::PLUS,
					std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
					std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}),
					SourceLocation{}),
				SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, FunctionDefinitionTypeAnnotation)
{
	constexpr std::string_view program =
		"def add(a: int, b: int) -> int:\n"
		"   return a + b\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("add",// function_name
		std::make_shared<Arguments>(
			std::vector<std::shared_ptr<Argument>>{
				std::make_shared<Argument>("a",
					std::make_shared<Name>("int", ContextType::LOAD, SourceLocation{}),
					"",
					SourceLocation{}),
				std::make_shared<Argument>("b",
					std::make_shared<Name>("int", ContextType::LOAD, SourceLocation{}),
					"",
					SourceLocation{}),
			},
			SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Return>(
				std::make_shared<BinaryExpr>(BinaryOpType::PLUS,
					std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
					std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}),
					SourceLocation{}),
				SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		std::make_shared<Name>("int", ContextType::LOAD, SourceLocation{}),// returns
		"",// type_comment
		SourceLocation{}));

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
		std::make_shared<Arguments>(
			std::vector<std::shared_ptr<Argument>>{
				std::make_shared<Argument>("a", nullptr, "", SourceLocation{}),
			},
			SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
										 "constant", ContextType::STORE, SourceLocation{}) },
				std::make_shared<Constant>(static_cast<int64_t>(1), SourceLocation{}),
				"",
				SourceLocation{}),
			std::make_shared<Return>(
				std::make_shared<BinaryExpr>(BinaryOpType::PLUS,
					std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
					std::make_shared<Name>("constant", ContextType::LOAD, SourceLocation{}),
					SourceLocation{}),
				SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, SimpleIfStatement)
{
	constexpr std::string_view program =
		"if True:\n"
		"   print(\"Hello, World!\")\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<If>(std::make_shared<Constant>(true, SourceLocation{}),// test
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
				std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>("Hello, World!", SourceLocation{}) },
				std::vector<std::shared_ptr<Keyword>>{},
				SourceLocation{}) },// body
			std::vector<std::shared_ptr<ASTNode>>{},// orelse
			SourceLocation{}));

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
	expected_ast->emplace(
		std::make_shared<If>(std::make_shared<Constant>(true, SourceLocation{}),// test
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
				std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>("Hello, World!", SourceLocation{}) },
				std::vector<std::shared_ptr<Keyword>>{},
				SourceLocation{}) },// body
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
				std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>("Goodbye!", SourceLocation{}) },
				std::vector<std::shared_ptr<Keyword>>{},
				SourceLocation{}) },// orelse
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, IfStatementWithComparisson)
{
	constexpr std::string_view program =
		"a = 1\n"
		"if a == 1:\n"
		"   a = 2\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Constant>(static_cast<int64_t>(1), SourceLocation{}),
			"",
			SourceLocation{}));
	expected_ast->emplace(std::make_shared<If>(
		std::make_shared<Compare>(std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
			std::vector{ Compare::OpType::Eq },
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(static_cast<int64_t>(1), SourceLocation{}) },
			SourceLocation{}),// test
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
										 "a", ContextType::STORE, SourceLocation{}) },
				std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}),
				"",
				SourceLocation{}) },// body
		std::vector<std::shared_ptr<ASTNode>>{},// orelse
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, LiteralList)
{
	constexpr std::string_view program = "a = [1, 2, 3, 5]\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<List>(
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
					std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}),
					std::make_shared<Constant>(int64_t{ 3 }, SourceLocation{}),
					std::make_shared<Constant>(int64_t{ 5 }, SourceLocation{}),
				},
				ContextType::LOAD,
				SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, LiteralTuple)
{
	constexpr std::string_view program = "a = (1, 2, 3, 5)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<ast::Tuple>(
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
					std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}),
					std::make_shared<Constant>(int64_t{ 3 }, SourceLocation{}),
					std::make_shared<Constant>(int64_t{ 5 }, SourceLocation{}),
				},
				ContextType::LOAD,
				SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, LiteralDict)
{
	constexpr std::string_view program = "a = {\"a\": 1, b:2}\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Dict>(
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>("a", SourceLocation{}),
					std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}),
				},
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
					std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}),
				},
				SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, LiteralSet)
{
	constexpr std::string_view program = "a = {1, 2, 3, 5}\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Set>(
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
					std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}),
					std::make_shared<Constant>(int64_t{ 3 }, SourceLocation{}),
					std::make_shared<Constant>(int64_t{ 5 }, SourceLocation{}),
				},
				ContextType::LOAD,
				SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SimpleForLoopWithFunctionCall)
{
	constexpr std::string_view program =
		"for x in range(10):\n"
		"	print(x)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<For>(
		std::make_shared<Name>("x", ContextType::STORE, SourceLocation{}),// target
		std::make_shared<Call>(std::make_shared<Name>("range", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 10 }, SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{},
			SourceLocation{}),// iter
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Name>("x", ContextType::LOAD, SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{},
			SourceLocation{}) },// body
		std::vector<std::shared_ptr<ASTNode>>{},// orelse
		"",// type_comment
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ForLoopMultipleTargets)
{
	constexpr std::string_view program =
		"for x, y in z:\n"
		"	print(x, y)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<For>(
		std::make_shared<ast::Tuple>(
			std::vector<std::shared_ptr<ast::ASTNode>>{
				std::make_shared<Name>("x", ContextType::STORE, SourceLocation{}),
				std::make_shared<Name>("y", ContextType::STORE, SourceLocation{}) },
			ContextType::STORE,
			SourceLocation{}),// target
		std::make_shared<Name>("z", ContextType::LOAD, SourceLocation{}),// iter
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Name>("x", ContextType::LOAD, SourceLocation{}),
				std::make_shared<Name>("y", ContextType::LOAD, SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{},
			SourceLocation{}) },// body
		std::vector<std::shared_ptr<ASTNode>>{},// orelse
		"",// type_comment
		SourceLocation{}));

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
		std::make_shared<Name>("x", ContextType::STORE, SourceLocation{}),// target
		std::make_shared<Call>(std::make_shared<Name>("range", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 10 }, SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{},
			SourceLocation{}),// iter
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Name>("x", ContextType::LOAD, SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{},
			SourceLocation{}) },// body
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Call>(
				std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>("ELSE!", SourceLocation{}) },
				std::vector<std::shared_ptr<Keyword>>{},
				SourceLocation{}),
		},// orelse
		"",// type_comment
		SourceLocation{}));

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
			std::make_shared<Name>("Base", ContextType::LOAD, SourceLocation{}) },// bases
		std::vector{ std::make_shared<Keyword>("metaclass",
			std::make_shared<Name>("MyMetaClass", ContextType::LOAD, SourceLocation{}),
			SourceLocation{}) },// Keywords
		std::vector<std::shared_ptr<ast::ASTNode>>{
			std::make_shared<FunctionDefinition>("__init__",// function_name
				std::make_shared<Arguments>(
					std::vector<std::shared_ptr<Argument>>{
						std::make_shared<Argument>("self", nullptr, "", SourceLocation{}),
						std::make_shared<Argument>("value", nullptr, "", SourceLocation{}),
					},
					SourceLocation{}),// args
				std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Assign>(
					std::vector<std::shared_ptr<ast::ASTNode>>{ std::make_shared<Attribute>(
						std::make_shared<Name>("self", ContextType::LOAD, SourceLocation{}),
						"value",
						ContextType::STORE,
						SourceLocation{}) },
					std::make_shared<Name>("value", ContextType::LOAD, SourceLocation{}),
					"",
					SourceLocation{}) },// body
				std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
				nullptr,// returns
				"",// type_comment
				SourceLocation{}) },// body
		std::vector<std::shared_ptr<ast::ASTNode>>{},// decorator_list
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AccessAttribute)
{
	constexpr std::string_view program = "test = foo.bar\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "test", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Attribute>(
				std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}),
				"bar",
				ContextType::LOAD,
				SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, CallMethod)
{
	constexpr std::string_view program = "test = foo.bar()\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "test", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Call>(
				std::make_shared<Attribute>(
					std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}),
					"bar",
					ContextType::LOAD,
					SourceLocation{}),
				SourceLocation{}),
			"",
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, LiteralMethodCall)
{
	constexpr std::string_view program = "\"foo123\".isalnum()\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(
		std::make_shared<Attribute>(std::make_shared<Constant>("foo123", SourceLocation{}),
			"isalnum",
			ContextType::LOAD,
			SourceLocation{}),
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionCallWithKwarg)
{
	constexpr std::string_view program = "print(\"Hello\", \"world!\", sep=',')\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>("Hello", SourceLocation{}),
				std::make_shared<Constant>("world!", SourceLocation{}) },
			std::vector{ std::make_shared<Keyword>(
				"sep", std::make_shared<Constant>(",", SourceLocation{}), SourceLocation{}) },
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionCallWithOnlyKwargs)
{
	constexpr std::string_view program = "add(a=1, b=2)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Call>(std::make_shared<Name>("add", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{},
			std::vector{ std::make_shared<Keyword>("a",
							 std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
							 SourceLocation{}),
				std::make_shared<Keyword>("b",
					std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}),
					SourceLocation{}) },
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionCallWithKwargAsResultFromAnotherFunction)
{
	constexpr std::string_view program = "print(\"Hello\", \"world!\", sep=my_separator())\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>("Hello", SourceLocation{}),
				std::make_shared<Constant>("world!", SourceLocation{}) },
			std::vector{ std::make_shared<Keyword>("sep",
				std::make_shared<Call>(
					std::make_shared<Name>("my_separator", ContextType::LOAD, SourceLocation{}),
					SourceLocation{}),
				SourceLocation{}) },
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}


TEST(Parser, AugmentedAssign)
{
	constexpr std::string_view program = "a += b\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<AugAssign>(
		std::make_shared<Name>("a", ContextType::STORE, SourceLocation{}),
		BinaryOpType::PLUS,
		std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}),
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AugmentedAssignOr)
{
	constexpr std::string_view program = "a |= b\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<AugAssign>(
		std::make_shared<Name>("a", ContextType::STORE, SourceLocation{}),
		BinaryOpType::OR,
		std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}),
		SourceLocation{}));

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
		std::make_shared<Compare>(std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
			std::vector{ Compare::OpType::LtE },
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 10 }, SourceLocation{}) },
			SourceLocation{}),
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<AugAssign>(
			std::make_shared<Name>("a", ContextType::STORE, SourceLocation{}),
			BinaryOpType::PLUS,
			std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
			SourceLocation{}) },
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{},
			SourceLocation{}) },
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, Import)
{
	constexpr std::string_view program = "import fibo\n";

	auto expected_ast = create_test_module();
	std::vector<alias> names{ alias{
		.name = "fibo",
	} };
	auto import = std::make_shared<Import>(std::move(names), SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportAs)
{
	constexpr std::string_view program = "import fibo as f\n";

	auto expected_ast = create_test_module();
	std::vector<alias> names{ alias{
		.name = "fibo",
		.asname = "f",
	} };
	auto import = std::make_shared<Import>(std::move(names), SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportDottedAs)
{
	constexpr std::string_view program = "import fibo.nac.ci as f\n";

	auto expected_ast = create_test_module();
	std::vector<alias> names{ alias{
		.name = "fibo.nac.ci",
		.asname = "f",
	} };
	auto import = std::make_shared<Import>(std::move(names), SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportMultiple)
{
	constexpr std::string_view program = "import fibo.nac.ci as f, bar as b, foo\n";

	auto expected_ast = create_test_module();

	std::vector<alias> names{
		alias{
			.name = "fibo.nac.ci",
			.asname = "f",
		},
		alias{
			.name = "bar",
			.asname = "b",
		},
		alias{
			.name = "foo",
		},
	};
	auto import = std::make_shared<Import>(std::move(names), SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportFrom)
{
	constexpr std::string_view program = "from sequence import fibo\n";

	auto expected_ast = create_test_module();
	std::vector<alias> names{
		alias{
			.name = "fibo",
		},
	};
	auto import = std::make_shared<ImportFrom>("sequence", std::move(names), 0, SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportFromDotted)
{
	constexpr std::string_view program = "from internal.math.sequence import fibo\n";

	auto expected_ast = create_test_module();
	std::vector<alias> names{ alias{
		.name = "fibo",
	} };
	auto import = std::make_shared<ImportFrom>(
		"internal.math.sequence", std::move(names), 0, SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportFromDottedMutiple)
{
	constexpr std::string_view program =
		"from internal.math.sequence import fibonacci as fib, factorial\n";

	auto expected_ast = create_test_module();

	std::vector<alias> names{
		alias{
			.name = "fibonacci",
			.asname = "fib",
		},
		alias{
			.name = "factorial",
		},
	};
	auto import = std::make_shared<ImportFrom>(
		"internal.math.sequence", std::move(names), 0, SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportFromDottedMutipleInParen)
{
	constexpr std::string_view program =
		"from internal.math.sequence import (fibonacci as fib, factorial)\n";

	auto expected_ast = create_test_module();

	std::vector<alias> names{
		alias{
			.name = "fibonacci",
			.asname = "fib",
		},
		alias{
			.name = "factorial",
		},
	};
	auto import = std::make_shared<ImportFrom>(
		"internal.math.sequence", std::move(names), 0, SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportFromParent)
{
	constexpr std::string_view program = "from .fibonacci.impl import fibonacci_cpu as fib\n";

	auto expected_ast = create_test_module();

	std::vector<alias> names{ alias{
		.name = "fibonacci_cpu",
		.asname = "fib",
	} };
	auto import =
		std::make_shared<ImportFrom>("fibonacci.impl", std::move(names), 1, SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportFromParentLevel2)
{
	constexpr std::string_view program = "from ..fibonacci.impl import fibonacci_cpu as fib\n";

	auto expected_ast = create_test_module();

	std::vector<alias> names{ alias{
		.name = "fibonacci_cpu",
		.asname = "fib",
	} };
	auto import =
		std::make_shared<ImportFrom>("fibonacci.impl", std::move(names), 2, SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportFromParentLevel3)
{
	constexpr std::string_view program = "from ...fibonacci.impl import fibonacci_cpu as fib\n";

	auto expected_ast = create_test_module();

	std::vector<alias> names{ alias{
		.name = "fibonacci_cpu",
		.asname = "fib",
	} };
	auto import =
		std::make_shared<ImportFrom>("fibonacci.impl", std::move(names), 3, SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ImportFromParentLevel4)
{
	constexpr std::string_view program = "from ....fibonacci.impl import fibonacci_cpu as fib\n";

	auto expected_ast = create_test_module();

	std::vector<alias> names{ alias{
		.name = "fibonacci_cpu",
		.asname = "fib",
	} };
	auto import =
		std::make_shared<ImportFrom>("fibonacci.impl", std::move(names), 4, SourceLocation{});
	expected_ast->emplace(import);

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SubscriptIndexExpression)
{
	constexpr std::string_view program = "a[0]\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Subscript>(
		std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
		Subscript::Index{ std::make_shared<Constant>(int64_t{ 0 }, SourceLocation{}) },
		ContextType::LOAD,
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, SubscriptIndexAssignment)
{
	constexpr std::string_view program = "a[0] = 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Subscript>(
			std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
			Subscript::Index{ std::make_shared<Constant>(int64_t{ 0 }, SourceLocation{}) },
			ContextType::STORE,
			SourceLocation{}) },
		std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
		"",
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SubscriptMultiIndexAssignment)
{
	constexpr std::string_view program = "a[0, 1, 2] = 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Subscript>(
			std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
			Subscript::ExtSlice{ .dims = { Subscript::Index{ .value = std::make_shared<Constant>(
																 int64_t{ 0 }, SourceLocation{}) },
									 Subscript::Index{ .value = std::make_shared<Constant>(
														   int64_t{ 1 }, SourceLocation{}) },
									 Subscript::Index{ .value = std::make_shared<Constant>(
														   int64_t{ 2 }, SourceLocation{}) } } },
			ContextType::STORE,
			SourceLocation{}) },
		std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
		"",
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SubscriptSliceExpression)
{
	constexpr std::string_view program =
		"a[0:10]\n"
		"b[0:10:2]\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Subscript>(
		std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
		Subscript::Slice{ .lower = std::make_shared<Constant>(int64_t{ 0 }, SourceLocation{}),
			.upper = std::make_shared<Constant>(int64_t{ 10 }, SourceLocation{}) },
		ContextType::LOAD,
		SourceLocation{}));
	expected_ast->emplace(std::make_shared<Subscript>(
		std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}),
		Subscript::Slice{ .lower = std::make_shared<Constant>(int64_t{ 0 }, SourceLocation{}),
			.upper = std::make_shared<Constant>(int64_t{ 10 }, SourceLocation{}),
			.step = std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}) },
		ContextType::LOAD,
		SourceLocation{}));


	assert_generates_ast(program, expected_ast);
}


TEST(Parser, SubscriptSliceExpressionNoStart)
{
	constexpr std::string_view program = "a[:2:1]\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Subscript>(
		std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
		Subscript::Slice{
			.lower = nullptr,
			.upper = std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}),
			.step = std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
		},
		ContextType::LOAD,
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SubscriptSliceExpressionNoStartOrEnd)
{
	constexpr std::string_view program = "a[::-1]\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Subscript>(
		std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
		Subscript::Slice{
			.lower = nullptr,
			.upper = nullptr,
			.step = std::make_shared<UnaryExpr>(UnaryOpType::SUB,
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
				SourceLocation{}),
		},
		ContextType::LOAD,
		SourceLocation{}));

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
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Subscript>(
			std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
			Subscript::Slice{ .lower = std::make_shared<Constant>(int64_t{ 0 }, SourceLocation{}),
				.upper = std::make_shared<Constant>(int64_t{ 10 }, SourceLocation{}) },
			ContextType::STORE,
			SourceLocation{}) },
		std::make_shared<Name>("c", ContextType::LOAD, SourceLocation{}),
		"",
		SourceLocation{}));
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Subscript>(
			std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}),
			Subscript::Slice{ .lower = std::make_shared<Constant>(int64_t{ 0 }, SourceLocation{}),
				.upper = std::make_shared<Constant>(int64_t{ 10 }, SourceLocation{}),
				.step = std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}) },
			ContextType::STORE,
			SourceLocation{}) },
		std::make_shared<Name>("d", ContextType::LOAD, SourceLocation{}),
		"",
		SourceLocation{}));
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Subscript>(
			std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}),
			Subscript::Slice{ .lower = std::make_shared<Constant>(int64_t{ 0 }, SourceLocation{}),
				.upper = std::make_shared<Name>("g", ContextType::LOAD, SourceLocation{}),
				.step = std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}) },
			ContextType::STORE,
			SourceLocation{}) },
		std::make_shared<Subscript>(
			std::make_shared<Name>("f", ContextType::LOAD, SourceLocation{}),
			Subscript::Slice{
				.lower = std::make_shared<Name>("e", ContextType::LOAD, SourceLocation{}),
				.upper = std::make_shared<Name>("h", ContextType::LOAD, SourceLocation{}) },
			ContextType::LOAD,
			SourceLocation{}),
		"",
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, Raise)
{
	constexpr std::string_view program = "raise\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Raise>(SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, RaiseValueError)
{
	constexpr std::string_view program = "raise ValueError(\"Wrong!\")\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Raise>(
		std::make_shared<Call>(
			std::make_shared<Name>("ValueError", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>("Wrong!", SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{},
			SourceLocation{}),
		nullptr,
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, RaiseValueErrorCause)
{
	constexpr std::string_view program = "raise ValueError(\"Wrong!\") from exc\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Raise>(
		std::make_shared<Call>(
			std::make_shared<Name>("ValueError", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>("Wrong!", SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{},
			SourceLocation{}),
		std::make_shared<Name>("exc", ContextType::LOAD, SourceLocation{}),
		SourceLocation{}));

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
	expected_ast->emplace(std::make_shared<Try>(
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}), SourceLocation{}) },
		std::vector<std::shared_ptr<ExceptHandler>>{},
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("bar", ContextType::LOAD, SourceLocation{}), SourceLocation{}) },
		SourceLocation{}));

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
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}), SourceLocation{}) },
		std::vector{ std::make_shared<ExceptHandler>(nullptr,
			"",
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
				std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>("Exception", SourceLocation{}) },
				std::vector<std::shared_ptr<Keyword>>{},
				SourceLocation{}) },
			SourceLocation{}) },
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<ASTNode>>{},
		SourceLocation{}));

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
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}), SourceLocation{}) },
		std::vector{ std::make_shared<ExceptHandler>(
			std::make_shared<Name>("ValueError", ContextType::LOAD, SourceLocation{}),
			"",
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
				std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>("Exception", SourceLocation{}) },
				std::vector<std::shared_ptr<Keyword>>{},
				SourceLocation{}) },
			SourceLocation{}) },
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<ASTNode>>{},
		SourceLocation{}));

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
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}), SourceLocation{}) },
		std::vector{ std::make_shared<ExceptHandler>(
			std::make_shared<Name>("ValueError", ContextType::LOAD, SourceLocation{}),
			"e",
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
				std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Name>("e", ContextType::LOAD, SourceLocation{}) },
				std::vector<std::shared_ptr<Keyword>>{},
				SourceLocation{}) },
			SourceLocation{}) },
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<ASTNode>>{},
		SourceLocation{}));

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
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}), SourceLocation{}) },
		std::vector{ std::make_shared<ExceptHandler>(
						 std::make_shared<Name>("ValueError", ContextType::LOAD, SourceLocation{}),
						 "e",
						 std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
							 std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
							 std::vector<std::shared_ptr<ASTNode>>{
								 std::make_shared<Name>("e", ContextType::LOAD, SourceLocation{}) },
							 std::vector<std::shared_ptr<Keyword>>{},
							 SourceLocation{}) },
						 SourceLocation{}),
			std::make_shared<ExceptHandler>(
				std::make_shared<Name>("BaseException", ContextType::LOAD, SourceLocation{}),
				"",
				std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
					std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
					std::vector<std::shared_ptr<ASTNode>>{
						std::make_shared<Constant>("BaseException", SourceLocation{}) },
					std::vector<std::shared_ptr<Keyword>>{},
					SourceLocation{}) },
				SourceLocation{}),
			std::make_shared<ExceptHandler>(nullptr,
				"",
				std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
					std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
					std::vector<std::shared_ptr<ASTNode>>{
						std::make_shared<Constant>("Exception", SourceLocation{}) },
					std::vector<std::shared_ptr<Keyword>>{},
					SourceLocation{}) },
				SourceLocation{}) },
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<ASTNode>>{},
		SourceLocation{}));

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
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}), SourceLocation{}) },
		std::vector{ std::make_shared<ExceptHandler>(
			std::make_shared<Name>("ValueError", ContextType::LOAD, SourceLocation{}),
			"e",
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
				std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Name>("e", ContextType::LOAD, SourceLocation{}) },
				std::vector<std::shared_ptr<Keyword>>{},
				SourceLocation{}) },
			SourceLocation{}) },
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Call>(
				std::make_shared<Name>("cleanup", ContextType::LOAD, SourceLocation{}),
				SourceLocation{}),
			std::make_shared<Call>(
				std::make_shared<Name>("exit", ContextType::LOAD, SourceLocation{}),
				SourceLocation{}) },
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, Assert)
{
	constexpr std::string_view program = "assert False\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assert>(
		std::make_shared<Constant>(false, SourceLocation{}), nullptr, SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AssertWithMessage)
{
	constexpr std::string_view program = "assert False, \"failed!\"\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assert>(std::make_shared<Constant>(false, SourceLocation{}),
			std::make_shared<Constant>("failed!", SourceLocation{}),
			SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, NegativeNumber)
{
	constexpr std::string_view program = "a = -1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<UnaryExpr>(UnaryOpType::SUB,
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
				SourceLocation{}),
			"",
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, UnaryExprMix)
{
	constexpr std::string_view program = "a = -+-+1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<UnaryExpr>(UnaryOpType::SUB,
				std::make_shared<UnaryExpr>(UnaryOpType::ADD,
					std::make_shared<UnaryExpr>(UnaryOpType::SUB,
						std::make_shared<UnaryExpr>(UnaryOpType::ADD,
							std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
							SourceLocation{}),
						SourceLocation{}),
					SourceLocation{}),
				SourceLocation{}),
			"",
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, UnaryNot)
{
	constexpr std::string_view program = "a = not b\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<UnaryExpr>(UnaryOpType::NOT,
				std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}),
				SourceLocation{}),
			"",
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, CompareIsNot)
{
	constexpr std::string_view program = "assert a is not False\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assert>(
		std::make_shared<Compare>(std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
			std::vector{ Compare::OpType::IsNot },
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(false, SourceLocation{}) },
			SourceLocation{}),
		nullptr,
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}


TEST(Parser, CompareIs)
{
	constexpr std::string_view program = "assert a is False\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assert>(
		std::make_shared<Compare>(std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
			std::vector{ Compare::OpType::Is },
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(false, SourceLocation{}) },
			SourceLocation{}),
		nullptr,
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, BoolOpAnd)
{
	constexpr std::string_view program = "a and b\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<BoolOp>(BoolOp::OpType::And,
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
			std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}) },
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, BoolOpOr)
{
	constexpr std::string_view program = "a or b\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<BoolOp>(BoolOp::OpType::Or,
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
			std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}) },
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, WithStatement)
{
	constexpr std::string_view program =
		"with lock:\n"
		"  work()\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<With>(
		std::vector{ std::make_shared<WithItem>(
			std::make_shared<Name>("lock", ContextType::LOAD, SourceLocation{}),
			nullptr,
			SourceLocation{}) },
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Call>(
			std::make_shared<Name>("work", ContextType::LOAD, SourceLocation{}),
			SourceLocation{}) },
		"",
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, IfExpression)
{
	constexpr std::string_view program = "a = 42 if magic() else 21\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<IfExpr>(
				std::make_shared<Call>(
					std::make_shared<Name>("magic", ContextType::LOAD, SourceLocation{}),
					SourceLocation{}),
				std::make_shared<Constant>(int64_t{ 42 }, SourceLocation{}),
				std::make_shared<Constant>(int64_t{ 21 }, SourceLocation{}),
				SourceLocation{}),
			"",
			SourceLocation{}));
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
			std::make_shared<Argument>("args", nullptr, "", SourceLocation{}),
			std::vector<std::shared_ptr<Argument>>{},
			std::vector<std::shared_ptr<ast::ASTNode>>{},
			std::make_shared<Argument>("kwargs", nullptr, "", SourceLocation{}),
			std::vector<std::shared_ptr<ast::ASTNode>>{},
			SourceLocation{}),
		std::vector<std::shared_ptr<ast::ASTNode>>{ std::make_shared<Return>(
			std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}), SourceLocation{}) },
		std::vector<std::shared_ptr<ast::ASTNode>>{},
		nullptr,
		"",
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ArgsKwargsFunctionCall)
{
	constexpr std::string_view program = "foo(*args, **kwargs)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Starred>(
				std::make_shared<Name>("args", ContextType::LOAD, SourceLocation{}),
				ContextType::LOAD,
				SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{ std::make_shared<Keyword>(
				std::make_shared<Name>("kwargs", ContextType::LOAD, SourceLocation{}),
				SourceLocation{}) },
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ArgKwargsFunctionCall)
{
	constexpr std::string_view program = "foo(arg, **kwargs)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Name>("arg", ContextType::LOAD, SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{ std::make_shared<Keyword>(
				std::make_shared<Name>("kwargs", ContextType::LOAD, SourceLocation{}),
				SourceLocation{}) },
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ArgsKwargsArgFunctionCall)
{
	constexpr std::string_view program = "foo(*args, **kwargs, a=1)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Call>(std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Starred>(
				std::make_shared<Name>("args", ContextType::LOAD, SourceLocation{}),
				ContextType::LOAD,
				SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{
				std::make_shared<Keyword>(
					std::make_shared<Name>("kwargs", ContextType::LOAD, SourceLocation{}),
					SourceLocation{}),
				std::make_shared<Keyword>("a",
					std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
					SourceLocation{}) },
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ComplexArgsKwargsFunctionCall)
{
	constexpr std::string_view program =
		"foo(*args, *more_args, bar, b=2, **kwargs, **more_kwargs, a=1)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(
		std::make_shared<Name>("foo", ContextType::LOAD, SourceLocation{}),
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Starred>(
				std::make_shared<Name>("args", ContextType::LOAD, SourceLocation{}),
				ContextType::LOAD,
				SourceLocation{}),
			std::make_shared<Starred>(
				std::make_shared<Name>("more_args", ContextType::LOAD, SourceLocation{}),
				ContextType::LOAD,
				SourceLocation{}),
			std::make_shared<Name>("bar", ContextType::LOAD, SourceLocation{}) },
		std::vector<std::shared_ptr<Keyword>>{
			std::make_shared<Keyword>(
				"b", std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}), SourceLocation{}),
			std::make_shared<Keyword>(
				std::make_shared<Name>("kwargs", ContextType::LOAD, SourceLocation{}),
				SourceLocation{}),
			std::make_shared<Keyword>(
				std::make_shared<Name>("more_kwargs", ContextType::LOAD, SourceLocation{}),
				SourceLocation{}),
			std::make_shared<Keyword>("a",
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
				SourceLocation{}) },
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AugAssignAttribute)
{
	constexpr std::string_view program = "a.b += 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<AugAssign>(
		std::make_shared<Attribute>(
			std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
			"b",
			ContextType::STORE,
			SourceLocation{}),
		BinaryOpType::PLUS,
		std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, AugAssignSlices)
{
	constexpr std::string_view program = "a['b'] += 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<AugAssign>(
		std::make_shared<Subscript>(
			std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}),
			Subscript::Index{ .value = std::make_shared<Constant>("b", SourceLocation{}) },
			ContextType::STORE,
			SourceLocation{}),
		BinaryOpType::PLUS,
		std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
		SourceLocation{}));
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
		std::make_shared<Arguments>(
			std::vector<std::shared_ptr<Argument>>{
				std::make_shared<Argument>("cls", nullptr, "", SourceLocation{}),
			},
			SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Return>(
				std::make_shared<Constant>(py::NameConstant{ py::NoneType{} }, SourceLocation{}),
				SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Name>("classmethod", ContextType::LOAD, SourceLocation{}),
			std::make_shared<Name>("_require_frozen", ContextType::LOAD, SourceLocation{}),
		},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));
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
			std::make_shared<Name>("Base", ContextType::LOAD, SourceLocation{}),
		},// bases
		std::vector<std::shared_ptr<Keyword>>{},// keyword
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Pass>(SourceLocation{}) },// body
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Name>("my_decorator", ContextType::LOAD, SourceLocation{}),
		},// decorator_list
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, NamedExpression)
{
	constexpr std::string_view program =
		"if a:=1:\n"
		"  pass\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<If>(std::make_shared<NamedExpr>(
								 std::make_shared<Name>("a", ContextType::STORE, SourceLocation{}),
								 std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
								 SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Pass>(SourceLocation{}) },
			std::vector<std::shared_ptr<ASTNode>>{},
			SourceLocation{}));
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
				std::make_shared<Argument>("a", nullptr, "", SourceLocation{}),
				std::make_shared<Argument>("b", nullptr, "", SourceLocation{}),
			},
			nullptr,
			std::vector<std::shared_ptr<ast::Argument>>{},
			std::vector<std::shared_ptr<ASTNode>>{},
			nullptr,
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}) },
			SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Pass>(SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));
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
				std::make_shared<Argument>("a", nullptr, "", SourceLocation{}),
			},
			nullptr,
			std::vector{ std::make_shared<ast::Argument>("b", nullptr, "", SourceLocation{}) },
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}) },
			nullptr,
			std::vector<std::shared_ptr<ASTNode>>{},
			SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Pass>(SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));
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
				std::make_shared<Argument>("a", nullptr, "", SourceLocation{}),
				std::make_shared<Argument>("b", nullptr, "", SourceLocation{}),
			},
			nullptr,
			std::vector{ std::make_shared<ast::Argument>("c", nullptr, "", SourceLocation{}) },
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}) },
			std::make_shared<Argument>("kwargs", nullptr, "", SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}) },
			SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Pass>(SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));
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
				std::make_shared<Argument>("a", nullptr, "", SourceLocation{}),
				std::make_shared<Argument>("b", nullptr, "", SourceLocation{}),
			},
			nullptr,
			std::vector{ std::make_shared<ast::Argument>("c", nullptr, "", SourceLocation{}),
				std::make_shared<ast::Argument>("d", nullptr, "", SourceLocation{}) },
			std::vector<std::shared_ptr<ASTNode>>{ nullptr, nullptr },
			std::make_shared<Argument>("kwargs", nullptr, "", SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}) },
			SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Pass>(SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionDefinitionWithPositionalArgsWithoutDefault)
{
	constexpr std::string_view program =
		"def f(a, /, b, **kwargs):\n"
		"  pass\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("f",// function_name
		std::make_shared<Arguments>(
			std::vector{
				std::make_shared<Argument>("a", nullptr, "", SourceLocation{}),
			},
			std::vector{
				std::make_shared<Argument>("b", nullptr, "", SourceLocation{}),
			},
			nullptr,
			std::vector<std::shared_ptr<Argument>>{},
			std::vector<std::shared_ptr<ASTNode>>{},
			std::make_shared<Argument>("kwargs", nullptr, "", SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{},
			SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Pass>(SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionDefinitionWithPositionalArgWithDefault)
{
	constexpr std::string_view program =
		"def f(a, b=(), /, c=1, **kwargs):\n"
		"  pass\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("f",// function_name
		std::make_shared<Arguments>(
			std::vector{
				std::make_shared<Argument>("a", nullptr, "", SourceLocation{}),
				std::make_shared<Argument>("b", nullptr, "", SourceLocation{}),
			},
			std::vector{
				std::make_shared<Argument>("c", nullptr, "", SourceLocation{}),
			},
			nullptr,
			std::vector<std::shared_ptr<Argument>>{},
			std::vector<std::shared_ptr<ASTNode>>{},
			std::make_shared<Argument>("kwargs", nullptr, "", SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<ast::Tuple>(
					std::vector<std::shared_ptr<ASTNode>>{}, ContextType::LOAD, SourceLocation{}),
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
			},
			SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Pass>(SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, FunctionDefinitionWithOnlyDefaultArguments)
{
	constexpr std::string_view program =
		"def f(a=None):\n"
		"  pass\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("f",// function_name
		std::make_shared<Arguments>(std::vector<std::shared_ptr<Argument>>{},
			std::vector{
				std::make_shared<Argument>("a", nullptr, "", SourceLocation{}),
			},
			nullptr,
			std::vector<std::shared_ptr<Argument>>{},
			std::vector<std::shared_ptr<ASTNode>>{},
			nullptr,
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(NameConstant{ NoneType{} }, SourceLocation{}),
			},
			SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Pass>(SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));
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
					std::make_shared<Name>("sys", ContextType::LOAD, SourceLocation{}),
					"modules",
					ContextType::LOAD,
					SourceLocation{}),
				"foo",
				ContextType::LOAD,
				SourceLocation{}),
			Subscript::Index{
				.value = std::make_shared<Attribute>(
					std::make_shared<Attribute>(
						std::make_shared<Name>("spec", ContextType::LOAD, SourceLocation{}),
						"name",
						ContextType::LOAD,
						SourceLocation{}),
					"bar",
					ContextType::LOAD,
					SourceLocation{}) },
			ContextType::STORE,
			SourceLocation{}) },
		std::make_shared<Attribute>(
			std::make_shared<Name>("my", ContextType::LOAD, SourceLocation{}),
			"module",
			ContextType::LOAD,
			SourceLocation{}),
		"",
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, CallAttributeSubscript)
{
	constexpr std::string_view program = "sys.modules.foo[spec.name.bar]()\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(
		std::make_shared<Subscript>(
			std::make_shared<Attribute>(
				std::make_shared<Attribute>(
					std::make_shared<Name>("sys", ContextType::LOAD, SourceLocation{}),
					"modules",
					ContextType::LOAD,
					SourceLocation{}),
				"foo",
				ContextType::LOAD,
				SourceLocation{}),
			Subscript::Index{
				.value = std::make_shared<Attribute>(
					std::make_shared<Attribute>(
						std::make_shared<Name>("spec", ContextType::LOAD, SourceLocation{}),
						"name",
						ContextType::LOAD,
						SourceLocation{}),
					"bar",
					ContextType::LOAD,
					SourceLocation{}) },
			ContextType::LOAD,
			SourceLocation{}),
		std::vector<std::shared_ptr<ASTNode>>{},
		std::vector<std::shared_ptr<Keyword>>{},
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, WithItemVar)
{
	constexpr std::string_view program =
		"with open(\"file.txt\") as f:\n"
		"  pass\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<With>(
		std::vector{ std::make_shared<WithItem>(
			std::make_shared<Call>(
				std::make_shared<Name>("open", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>("file.txt", SourceLocation{}) },
				std::vector<std::shared_ptr<Keyword>>{},
				SourceLocation{}),
			std::make_shared<Name>("f", ContextType::STORE, SourceLocation{}),
			SourceLocation{}) },
		std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Pass>(SourceLocation{}) },
		"",
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ListComprehension)
{
	constexpr std::string_view program = "[len(sep) == 1 for sep in path_separators]\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<ListComp>(
		std::make_shared<Compare>(
			std::make_shared<Call>(
				std::make_shared<Name>("len", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ast::ASTNode>>{
					std::make_shared<Name>("sep", ContextType::LOAD, SourceLocation{}) },
				std::vector<std::shared_ptr<ast::Keyword>>{},
				SourceLocation{}),
			std::vector{ Compare::OpType::Eq },
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}) },
			SourceLocation{}),
		std::vector<std::shared_ptr<Comprehension>>{
			std::make_shared<Comprehension>(
				std::make_shared<Name>("sep", ContextType::STORE, SourceLocation{}),
				std::make_shared<Name>("path_separators", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ASTNode>>{},
				false,
				SourceLocation{}),
		},
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, ListComprehensionIf)
{
	constexpr std::string_view program = "[sep for sep in path_separators if len(sep) == 1]\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<ListComp>(
		std::make_shared<Name>("sep", ContextType::LOAD, SourceLocation{}),
		std::vector<std::shared_ptr<Comprehension>>{
			std::make_shared<Comprehension>(
				std::make_shared<Name>("sep", ContextType::STORE, SourceLocation{}),
				std::make_shared<Name>("path_separators", ContextType::LOAD, SourceLocation{}),
				std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Compare>(
					std::make_shared<Call>(
						std::make_shared<Name>("len", ContextType::LOAD, SourceLocation{}),
						std::vector<std::shared_ptr<ast::ASTNode>>{
							std::make_shared<Name>("sep", ContextType::LOAD, SourceLocation{}) },
						std::vector<std::shared_ptr<ast::Keyword>>{},
						SourceLocation{}),
					std::vector{ Compare::OpType::Eq },
					std::vector<std::shared_ptr<ASTNode>>{
						std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}) },
					SourceLocation{}) },
				false,
				SourceLocation{}),
		},
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, DictComprehension)
{
	constexpr std::string_view program = "{k: v for k, v in container}\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<DictComp>(std::make_shared<Name>("k", ContextType::LOAD, SourceLocation{}),
			std::make_shared<Name>("v", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<Comprehension>>{
				std::make_shared<Comprehension>(
					std::make_shared<ast::Tuple>(
						std::vector<std::shared_ptr<ASTNode>>{
							std::make_shared<Name>("k", ContextType::STORE, SourceLocation{}),
							std::make_shared<Name>("v", ContextType::STORE, SourceLocation{}),
						},
						ContextType::STORE,
						SourceLocation{}),
					std::make_shared<Name>("container", ContextType::LOAD, SourceLocation{}),
					std::vector<std::shared_ptr<ASTNode>>{},
					false,
					SourceLocation{}),
			},
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}


TEST(Parser, GeneratorExpr)
{
	constexpr std::string_view program = "all(len(sep) == 1 for sep in path_separators)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Call>(
		std::make_shared<Name>("all", ContextType::LOAD, SourceLocation{}),
		std::vector<std::shared_ptr<ast::ASTNode>>{ std::make_shared<GeneratorExp>(
			std::make_shared<Compare>(
				std::make_shared<Call>(
					std::make_shared<Name>("len", ContextType::LOAD, SourceLocation{}),
					std::vector<std::shared_ptr<ast::ASTNode>>{
						std::make_shared<Name>("sep", ContextType::LOAD, SourceLocation{}) },
					std::vector<std::shared_ptr<ast::Keyword>>{},
					SourceLocation{}),
				std::vector{ Compare::OpType::Eq },
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}) },
				SourceLocation{}),
			std::vector<std::shared_ptr<Comprehension>>{
				std::make_shared<Comprehension>(
					std::make_shared<Name>("sep", ContextType::STORE, SourceLocation{}),
					std::make_shared<Name>("path_separators", ContextType::LOAD, SourceLocation{}),
					std::vector<std::shared_ptr<ASTNode>>{},
					false,
					SourceLocation{}),
			},
			SourceLocation{}) },
		std::vector<std::shared_ptr<ast::Keyword>>{},
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SetComp)
{
	constexpr std::string_view program = "{s for s in path_separators}\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<SetComp>(std::make_shared<Name>("s", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<Comprehension>>{
				std::make_shared<Comprehension>(
					std::make_shared<Name>("s", ContextType::STORE, SourceLocation{}),
					std::make_shared<Name>("path_separators", ContextType::LOAD, SourceLocation{}),
					std::vector<std::shared_ptr<ASTNode>>{},
					false,
					SourceLocation{}),
			},
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, Bytes)
{
	constexpr std::string_view program = "b\"hello\"\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Constant>(Bytes{ { std::byte{ 'h' },
														 std::byte{ 'e' },
														 std::byte{ 'l' },
														 std::byte{ 'l' },
														 std::byte{ 'o' } } },
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, LambdaWithNoArgs)
{
	constexpr std::string_view program = "a = lambda: 1\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Name>("a", ContextType::STORE, SourceLocation{}),
		},
		std::make_shared<Lambda>(
			std::make_shared<Arguments>(
				std::vector<std::shared_ptr<Argument>>{}, SourceLocation{}),// args
			std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
			SourceLocation{}),
		"",
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, LambdaWithNoDefaultArg)
{
	constexpr std::string_view program = "a = lambda x: x\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<Assign>(
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Name>("a", ContextType::STORE, SourceLocation{}),
		},
		std::make_shared<Lambda>(
			std::make_shared<Arguments>(
				std::vector<std::shared_ptr<Argument>>{
					std::make_shared<Argument>("x", nullptr, "", SourceLocation{}),
				},
				SourceLocation{}),// args
			std::make_shared<Name>("x", ContextType::LOAD, SourceLocation{}),
			SourceLocation{}),
		"",
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, Yield)
{
	constexpr std::string_view program =
		"def gen():\n"
		"   yield 1\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("gen",// function_name
		std::make_shared<Arguments>(
			std::vector<std::shared_ptr<Argument>>{}, SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Yield>(
				std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}), SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, YieldEmpty)
{
	constexpr std::string_view program =
		"def gen():\n"
		"   yield\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("gen",// function_name
		std::make_shared<Arguments>(
			std::vector<std::shared_ptr<Argument>>{}, SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Yield>(
				std::make_shared<Constant>(NameConstant{ NoneType{} }, SourceLocation{}),
				SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, YieldMutipleValues)
{
	constexpr std::string_view program =
		"def gen():\n"
		"   yield 1, 2\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("gen",// function_name
		std::make_shared<Arguments>(
			std::vector<std::shared_ptr<Argument>>{}, SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<Yield>(
				std::make_shared<ast::Tuple>(
					std::vector<std::shared_ptr<ASTNode>>{
						std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
						std::make_shared<Constant>(int64_t{ 2 }, SourceLocation{}),
					},
					ContextType::LOAD,
					SourceLocation{}),
				SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}

TEST(Parser, YieldFrom)
{
	constexpr std::string_view program =
		"def foo():\n"
		"   yield from bar\n";
	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<FunctionDefinition>("foo",// function_name
		std::make_shared<Arguments>(
			std::vector<std::shared_ptr<Argument>>{}, SourceLocation{}),// args
		std::vector<std::shared_ptr<ASTNode>>{
			std::make_shared<YieldFrom>(
				std::make_shared<Name>("bar", ContextType::LOAD, SourceLocation{}),
				SourceLocation{}),
		},// body
		std::vector<std::shared_ptr<ASTNode>>{},// decorator_list
		nullptr,// returns
		"",// type_comment
		SourceLocation{}));

	assert_generates_ast(program, expected_ast);
}


TEST(Parser, Coroutine)
{
	constexpr std::string_view program =
		"async def foo():\n"
		"  return 1\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(std::make_shared<AsyncFunctionDefinition>("foo",
		std::make_shared<Arguments>(std::vector<std::shared_ptr<Argument>>{}, SourceLocation{}),
		std::vector<std::shared_ptr<ast::ASTNode>>{ std::make_shared<Return>(
			std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}), SourceLocation{}) },
		std::vector<std::shared_ptr<ast::ASTNode>>{},
		nullptr,
		"",
		SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

TEST(Parser, SemiColon)
{
	constexpr std::string_view program = "a = 1; print(a)\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Constant>(int64_t{ 1 }, SourceLocation{}),
			"",
			SourceLocation{}));
	expected_ast->emplace(
		std::make_shared<Call>(std::make_shared<Name>("print", ContextType::LOAD, SourceLocation{}),
			std::vector<std::shared_ptr<ASTNode>>{
				std::make_shared<Name>("a", ContextType::LOAD, SourceLocation{}) },
			std::vector<std::shared_ptr<Keyword>>{},
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}


TEST(Parser, UnpackKVPair)
{
	constexpr std::string_view program = "a = {**b, 'foo': 'bar', **c}\n";

	auto expected_ast = create_test_module();
	expected_ast->emplace(
		std::make_shared<Assign>(std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Name>(
									 "a", ContextType::STORE, SourceLocation{}) },
			std::make_shared<Dict>(
				std::vector<std::shared_ptr<ASTNode>>{
					nullptr,
					std::make_shared<Constant>("foo", SourceLocation{}),
					nullptr,
				},
				std::vector<std::shared_ptr<ASTNode>>{
					std::make_shared<Name>("b", ContextType::LOAD, SourceLocation{}),
					std::make_shared<Constant>("bar", SourceLocation{}),
					std::make_shared<Name>("c", ContextType::LOAD, SourceLocation{}),
				},
				SourceLocation{}),
			"",
			SourceLocation{}));
	assert_generates_ast(program, expected_ast);
}

// TEST(Parser, FString)
// {
// 	constexpr std::string_view program = "f\"sin({a}) is {sin(a):.3}\"\n";

// 	auto expected_ast = create_test_module();
// 	expected_ast->emplace(std::make_shared<JoinedStr>(std::vector<std::shared_ptr<ASTNode>>{
// 		std::make_shared<Constant>("sin("),
// 		std::make_shared<FormattedValue>(std::make_shared<Name>("a", ContextType::LOAD),
// 			FormattedValue::Conversion::NONE,
// 			nullptr),
// 		std::make_shared<Constant>(") is "),
// 		std::make_shared<FormattedValue>(
// 			std::make_shared<Call>(std::make_shared<Name>("sin", ContextType::LOAD),
// 				std::vector<std::shared_ptr<ast::ASTNode>>{
// 					std::make_shared<Name>("a", ContextType::LOAD) },
// 				std::vector<std::shared_ptr<ast::Keyword>>{}),
// 			FormattedValue::Conversion::NONE,
// 			std::make_shared<JoinedStr>(
// 				std::vector<std::shared_ptr<ASTNode>>{ std::make_shared<Constant>(".3") }))
// 	}));
// 	assert_generates_ast(program, expected_ast);
// }
