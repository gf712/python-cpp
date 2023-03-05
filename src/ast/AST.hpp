#pragma once

#include <memory>
#include <numeric>
#include <optional>
#include <stack>
#include <string>
#include <variant>
#include <vector>

#include <gmpxx.h>

#include "forward.hpp"
#include "lexer/Lexer.hpp"
#include "utilities.hpp"

#include "spdlog/spdlog.h"

struct SourceLocation
{
	Position start;
	Position end;
};

namespace ast {

#define AST_NODE_TYPES                       \
	__AST_NODE_TYPE(Argument)                \
	__AST_NODE_TYPE(Arguments)               \
	__AST_NODE_TYPE(Attribute)               \
	__AST_NODE_TYPE(Assign)                  \
	__AST_NODE_TYPE(Assert)                  \
	__AST_NODE_TYPE(AsyncFunctionDefinition) \
	__AST_NODE_TYPE(AugAssign)               \
	__AST_NODE_TYPE(Break)                   \
	__AST_NODE_TYPE(BinaryExpr)              \
	__AST_NODE_TYPE(BoolOp)                  \
	__AST_NODE_TYPE(Call)                    \
	__AST_NODE_TYPE(ClassDefinition)         \
	__AST_NODE_TYPE(Continue)                \
	__AST_NODE_TYPE(Compare)                 \
	__AST_NODE_TYPE(Comprehension)           \
	__AST_NODE_TYPE(Constant)                \
	__AST_NODE_TYPE(Delete)                  \
	__AST_NODE_TYPE(Dict)                    \
	__AST_NODE_TYPE(DictComp)                \
	__AST_NODE_TYPE(ExceptHandler)           \
	__AST_NODE_TYPE(Expression)              \
	__AST_NODE_TYPE(For)                     \
	__AST_NODE_TYPE(FormattedValue)          \
	__AST_NODE_TYPE(FunctionDefinition)      \
	__AST_NODE_TYPE(GeneratorExp)            \
	__AST_NODE_TYPE(Global)                  \
	__AST_NODE_TYPE(If)                      \
	__AST_NODE_TYPE(IfExpr)                  \
	__AST_NODE_TYPE(Import)                  \
	__AST_NODE_TYPE(ImportFrom)              \
	__AST_NODE_TYPE(JoinedStr)               \
	__AST_NODE_TYPE(Keyword)                 \
	__AST_NODE_TYPE(Lambda)                  \
	__AST_NODE_TYPE(List)                    \
	__AST_NODE_TYPE(ListComp)                \
	__AST_NODE_TYPE(Module)                  \
	__AST_NODE_TYPE(NamedExpr)               \
	__AST_NODE_TYPE(Name)                    \
	__AST_NODE_TYPE(NonLocal)                \
	__AST_NODE_TYPE(Pass)                    \
	__AST_NODE_TYPE(Raise)                   \
	__AST_NODE_TYPE(Return)                  \
	__AST_NODE_TYPE(Set)                     \
	__AST_NODE_TYPE(SetComp)                 \
	__AST_NODE_TYPE(Starred)                 \
	__AST_NODE_TYPE(Subscript)               \
	__AST_NODE_TYPE(Try)                     \
	__AST_NODE_TYPE(Tuple)                   \
	__AST_NODE_TYPE(UnaryExpr)               \
	__AST_NODE_TYPE(While)                   \
	__AST_NODE_TYPE(With)                    \
	__AST_NODE_TYPE(WithItem)                \
	__AST_NODE_TYPE(Yield)                   \
	__AST_NODE_TYPE(YieldFrom)

class Value
{
	std::string m_name;

  public:
	Value(const std::string &name) : m_name(name) {}
	virtual ~Value() = default;
	const std::string &get_name() const { return m_name; }
};

enum class ASTNodeType {
#define __AST_NODE_TYPE(x) x,
	AST_NODE_TYPES
#undef __AST_NODE_TYPE
};

inline std::string_view node_type_to_string(ASTNodeType node_type)
{
	switch (node_type) {
#define __AST_NODE_TYPE(x) \
	case ASTNodeType::x:   \
		return #x;
		AST_NODE_TYPES
#undef __AST_NODE_TYPE
	}
	ASSERT_NOT_REACHED();
}

#define __AST_NODE_TYPE(x) class x;
AST_NODE_TYPES
#undef __AST_NODE_TYPE

enum class ContextType { LOAD = 0, STORE = 1, DELETE = 2, UNSET = 3 };

struct CodeGenerator;

class ASTContext
{
	std::stack<std::shared_ptr<Arguments>> m_local_args;
	std::vector<const ASTNode *> m_parent_nodes;

  public:
	void push_local_args(std::shared_ptr<Arguments> args) { m_local_args.push(std::move(args)); }
	void pop_local_args() { m_local_args.pop(); }

	bool has_local_args() const { return !m_local_args.empty(); }

	void push_node(const ASTNode *node) { m_parent_nodes.push_back(node); }
	void pop_node() { m_parent_nodes.pop_back(); }

	const std::shared_ptr<Arguments> &local_args() const { return m_local_args.top(); }
	const std::vector<const ASTNode *> &parent_nodes() const { return m_parent_nodes; }
};

class ASTNode
{
	const ASTNodeType m_node_type;
	const SourceLocation m_source_location;

  private:
	virtual void print_this_node(const std::string &indent) const = 0;
	virtual void dump() const { print_this_node(""); }

  public:
	ASTNode(ASTNodeType node_type, SourceLocation source_location)
		: m_node_type(node_type), m_source_location(source_location)
	{}
	void print_node(const std::string &indent) { print_this_node(indent); }
	ASTNodeType node_type() const { return m_node_type; }
	virtual ~ASTNode() = default;

	const SourceLocation &source_location() const { return m_source_location; }

	virtual Value *codegen(CodeGenerator *) const = 0;
};// namespace ast

class Expression : public ASTNode
{
	std::shared_ptr<ASTNode> m_value;

  public:
	Expression(std::shared_ptr<ASTNode> value, SourceLocation source_location)
		: ASTNode(ASTNodeType::Expression, source_location), m_value(std::move(value))
	{}

	const std::shared_ptr<ASTNode> &value() const { return m_value; }

	Value *codegen(CodeGenerator *) const override;

	void print_this_node(const std::string &indent) const override;
};

class Constant : public ASTNode
{
	std::unique_ptr<py::Value> m_value;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Constant(double value, SourceLocation source_location);
	Constant(int64_t value, SourceLocation source_location);
	Constant(mpz_class value, SourceLocation source_location);
	Constant(bool value, SourceLocation source_location);
	Constant(std::string value, SourceLocation source_location);
	Constant(const char *value, SourceLocation source_location);
	Constant(const py::Value &, SourceLocation source_location);

	const py::Value *value() const { return m_value.get(); }

	Value *codegen(CodeGenerator *) const override;
};


class List : public ASTNode
{
  private:
	std::vector<std::shared_ptr<ASTNode>> m_elements;
	ContextType m_ctx;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	List(std::vector<std::shared_ptr<ASTNode>> elements,
		ContextType ctx,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::List, source_location), m_elements(std::move(elements)), m_ctx(ctx)
	{}

	List(ContextType ctx, SourceLocation source_location)
		: ASTNode(ASTNodeType::List, source_location), m_elements(), m_ctx(ctx)
	{}

	void append(std::shared_ptr<ASTNode> element) { m_elements.push_back(std::move(element)); }

	ContextType context() const { return m_ctx; }
	const std::vector<std::shared_ptr<ASTNode>> &elements() const { return m_elements; }

	Value *codegen(CodeGenerator *) const override;
};

class Tuple : public ASTNode
{
  private:
	std::vector<std::shared_ptr<ASTNode>> m_elements;
	ContextType m_ctx;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Tuple(std::vector<std::shared_ptr<ASTNode>> elements,
		ContextType ctx,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Tuple, source_location), m_elements(std::move(elements)), m_ctx(ctx)
	{}

	Tuple(ContextType ctx, SourceLocation source_location)
		: ASTNode(ASTNodeType::Tuple, source_location), m_elements(), m_ctx(ctx)
	{}

	void append(std::shared_ptr<ASTNode> element) { m_elements.push_back(std::move(element)); }

	ContextType context() const { return m_ctx; }
	const std::vector<std::shared_ptr<ASTNode>> &elements() const { return m_elements; }
	std::vector<std::shared_ptr<ASTNode>> &elements() { return m_elements; }

	Value *codegen(CodeGenerator *) const override;
};


class Dict : public ASTNode
{
  private:
	std::vector<std::shared_ptr<ASTNode>> m_keys;
	std::vector<std::shared_ptr<ASTNode>> m_values;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Dict(std::vector<std::shared_ptr<ASTNode>> keys,
		std::vector<std::shared_ptr<ASTNode>> values,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Dict, source_location), m_keys(std::move(keys)),
		  m_values(std::move(values))
	{}

	Dict(SourceLocation source_location)
		: ASTNode(ASTNodeType::Dict, source_location), m_keys(), m_values()
	{}

	const std::vector<std::shared_ptr<ASTNode>> &keys() const { return m_keys; }
	const std::vector<std::shared_ptr<ASTNode>> &values() const { return m_values; }

	Value *codegen(CodeGenerator *) const override;
};


class Set : public ASTNode
{
  private:
	std::vector<std::shared_ptr<ASTNode>> m_elements;
	ContextType m_ctx;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Set(std::vector<std::shared_ptr<ASTNode>> elements,
		ContextType ctx,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::List, source_location), m_elements(std::move(elements)), m_ctx(ctx)
	{}

	ContextType context() const { return m_ctx; }
	const std::vector<std::shared_ptr<ASTNode>> &elements() const { return m_elements; }

	Value *codegen(CodeGenerator *) const override;
};


class Variable : public ASTNode
{
  public:
	ContextType context_type() const { return m_ctx; }

	virtual const std::vector<std::string> &ids() const = 0;

  protected:
	ContextType m_ctx;

  protected:
	Variable(ASTNodeType node_type, ContextType ctx, SourceLocation source_location)
		: ASTNode(node_type, source_location), m_ctx(ctx)
	{}
};


class Name final : public Variable
{
	std::vector<std::string> m_id;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Name(std::string id, ContextType ctx, SourceLocation source_location)
		: Variable(ASTNodeType::Name, ctx, source_location), m_id({ std::move(id) })
	{}

	const std::vector<std::string> &ids() const final { return m_id; }
	void set_context(ContextType ctx) { m_ctx = ctx; }

	Value *codegen(CodeGenerator *) const override;
};


class Statement : public ASTNode
{
  public:
	Statement(ASTNodeType node_type, SourceLocation source_location)
		: ASTNode(node_type, source_location)
	{}
};

class Assign : public Statement
{
	std::vector<std::shared_ptr<ASTNode>> m_targets;
	std::shared_ptr<ASTNode> m_value;
	std::string m_type_comment;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Assign(std::vector<std::shared_ptr<ASTNode>> targets,
		std::shared_ptr<ASTNode> value,
		std::string type_comment,
		SourceLocation source_location)
		: Statement(ASTNodeType::Assign, source_location), m_targets(std::move(targets)),
		  m_value(std::move(value)), m_type_comment(std::move(type_comment))
	{}

	const std::vector<std::shared_ptr<ASTNode>> &targets() const { return m_targets; }
	const std::shared_ptr<ASTNode> &value() const { return m_value; }
	void set_value(std::shared_ptr<ASTNode> v) { m_value = std::move(v); }

	Value *codegen(CodeGenerator *) const override;
};

#define UNARY_OPERATIONS \
	__UNARY_OP(ADD)      \
	__UNARY_OP(SUB)      \
	__UNARY_OP(NOT)      \
	__UNARY_OP(INVERT)

enum class UnaryOpType {
#define __UNARY_OP(x) x,
	UNARY_OPERATIONS
#undef __UNARY_OP
};

inline std::string_view stringify_unary_op(UnaryOpType op)
{
	switch (op) {
#define __UNARY_OP(x)    \
	case UnaryOpType::x: \
		return #x;
		UNARY_OPERATIONS
#undef __UNARY_OP
	}
	ASSERT_NOT_REACHED();
}

class UnaryExpr : public ASTNode
{
  public:
  private:
	const UnaryOpType m_op_type;
	std::shared_ptr<ASTNode> m_operand;

  public:
	UnaryExpr(UnaryOpType op_type, std::shared_ptr<ASTNode> operand, SourceLocation source_location)
		: ASTNode(ASTNodeType::UnaryExpr, source_location), m_op_type(op_type),
		  m_operand(std::move(operand))
	{}

	const std::shared_ptr<ASTNode> &operand() const { return m_operand; }
	std::shared_ptr<ASTNode> &operand() { return m_operand; }

	UnaryOpType op_type() const { return m_op_type; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

#define BINARY_OPERATIONS   \
	__BINARY_OP(PLUS)       \
	__BINARY_OP(MINUS)      \
	__BINARY_OP(MODULO)     \
	__BINARY_OP(MULTIPLY)   \
	__BINARY_OP(EXP)        \
	__BINARY_OP(SLASH)      \
	__BINARY_OP(FLOORDIV)   \
	__BINARY_OP(MATMUL)     \
	__BINARY_OP(LEFTSHIFT)  \
	__BINARY_OP(RIGHTSHIFT) \
	__BINARY_OP(AND)        \
	__BINARY_OP(OR)         \
	__BINARY_OP(XOR)

enum class BinaryOpType {
#define __BINARY_OP(x) x,
	BINARY_OPERATIONS
#undef __BINARY_OP
};

inline std::string_view stringify_binary_op(BinaryOpType op)
{
	switch (op) {
#define __BINARY_OP(x)    \
	case BinaryOpType::x: \
		return #x;
		BINARY_OPERATIONS
#undef __BINARY_OP
	}
	ASSERT_NOT_REACHED();
}

class BinaryExpr : public ASTNode
{
  public:
  private:
	const BinaryOpType m_op_type;
	std::shared_ptr<ASTNode> m_lhs;
	std::shared_ptr<ASTNode> m_rhs;

  public:
	BinaryExpr(BinaryOpType op_type,
		std::shared_ptr<ASTNode> lhs,
		std::shared_ptr<ASTNode> rhs,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::BinaryExpr, source_location), m_op_type(op_type),
		  m_lhs(std::move(lhs)), m_rhs(std::move(rhs))
	{}

	const std::shared_ptr<ASTNode> &lhs() const { return m_lhs; }
	std::shared_ptr<ASTNode> &lhs() { return m_lhs; }

	const std::shared_ptr<ASTNode> &rhs() const { return m_rhs; }
	std::shared_ptr<ASTNode> &rhs() { return m_rhs; }

	BinaryOpType op_type() const { return m_op_type; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class AugAssign : public Statement
{
	std::shared_ptr<ASTNode> m_target;
	BinaryOpType m_op;
	std::shared_ptr<ASTNode> m_value;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	AugAssign(std::shared_ptr<ASTNode> target,
		BinaryOpType op,
		std::shared_ptr<ASTNode> value,
		SourceLocation source_location)
		: Statement(ASTNodeType::AugAssign, source_location), m_target(std::move(target)), m_op(op),
		  m_value(std::move(value))
	{}

	const std::shared_ptr<ASTNode> &target() const { return m_target; }
	BinaryOpType op() const { return m_op; }
	const std::shared_ptr<ASTNode> &value() const { return m_value; }
	void set_value(std::shared_ptr<ASTNode> value) { m_value = std::move(value); }

	Value *codegen(CodeGenerator *) const override;
};

class Return : public ASTNode
{
	std::shared_ptr<ASTNode> m_value;

  public:
	Return(std::shared_ptr<ASTNode> value, SourceLocation source_location)
		: ASTNode(ASTNodeType::Return, source_location), m_value(std::move(value))
	{}

	std::shared_ptr<ASTNode> value() const { return m_value; }

	void print_this_node(const std::string &indent) const override;

	Value *codegen(CodeGenerator *) const override;
};

class Yield : public ASTNode
{
	std::shared_ptr<ASTNode> m_value;

  public:
	Yield(std::shared_ptr<ASTNode> value, SourceLocation source_location)
		: ASTNode(ASTNodeType::Yield, source_location), m_value(std::move(value))
	{}

	std::shared_ptr<ASTNode> value() const { return m_value; }

	void print_this_node(const std::string &indent) const override;

	Value *codegen(CodeGenerator *) const override;
};

class YieldFrom : public ASTNode
{
	std::shared_ptr<ASTNode> m_value;

  public:
	YieldFrom(std::shared_ptr<ASTNode> value, SourceLocation source_location)
		: ASTNode(ASTNodeType::YieldFrom, source_location), m_value(std::move(value))
	{}

	std::shared_ptr<ASTNode> value() const { return m_value; }

	void print_this_node(const std::string &indent) const override;

	Value *codegen(CodeGenerator *) const override;
};

class Argument final : public ASTNode
{
	const std::string m_arg;
	const std::shared_ptr<ASTNode> m_annotation;
	const std::string m_type_comment;

  public:
	Argument(std::string arg,
		std::shared_ptr<ASTNode> annotation,
		std::string type_comment,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Argument, source_location), m_arg(std::move(arg)),
		  m_annotation(std::move(annotation)), m_type_comment(std::move(type_comment))
	{}

	void print_this_node(const std::string &indent) const final;

	const std::string &name() const { return m_arg; }
	const std::shared_ptr<ASTNode> &annotation() const { return m_annotation; }

	Value *codegen(CodeGenerator *) const override;
};


class Arguments : public ASTNode
{
	std::vector<std::shared_ptr<Argument>> m_posonlyargs;
	std::vector<std::shared_ptr<Argument>> m_args;
	std::shared_ptr<Argument> m_vararg;
	std::vector<std::shared_ptr<Argument>> m_kwonlyargs;
	std::vector<std::shared_ptr<ASTNode>> m_kw_defaults;
	std::shared_ptr<Argument> m_kwarg;
	std::vector<std::shared_ptr<ASTNode>> m_defaults;

  public:
	Arguments(SourceLocation source_location) : ASTNode(ASTNodeType::Arguments, source_location) {}
	Arguments(std::vector<std::shared_ptr<Argument>> args, SourceLocation source_location)
		: Arguments(source_location)
	{
		m_args = std::move(args);
	}

	Arguments(std::vector<std::shared_ptr<Argument>> posonlyargs,
		std::vector<std::shared_ptr<Argument>> args,
		std::shared_ptr<Argument> vararg,
		std::vector<std::shared_ptr<Argument>> kwonlyargs,
		std::vector<std::shared_ptr<ASTNode>> kw_defaults,
		std::shared_ptr<Argument> kwarg,
		std::vector<std::shared_ptr<ASTNode>> defaults,
		SourceLocation source_location)
		: Arguments(source_location)
	{
		m_posonlyargs = std::move(posonlyargs);
		m_args = std::move(args);
		m_vararg = std::move(vararg);
		m_kwonlyargs = std::move(kwonlyargs);
		m_kw_defaults = std::move(kw_defaults);
		m_kwarg = std::move(kwarg);
		m_defaults = std::move(defaults);
	}

	void print_this_node(const std::string &indent) const final;

	void push_positional_arg(std::shared_ptr<Argument> arg)
	{
		m_posonlyargs.push_back(std::move(arg));
	}

	void push_arg(std::shared_ptr<Argument> arg) { m_args.push_back(std::move(arg)); }

	std::vector<std::string> argument_names() const;

	std::vector<std::string> kw_only_argument_names() const;

	void push_kwonlyarg(std::shared_ptr<Argument> kwarg)
	{
		m_kwonlyargs.push_back(std::move(kwarg));
	}
	std::vector<std::string> keyword_argument_names() const;

	void push_default(std::shared_ptr<ASTNode> default_value)
	{
		m_defaults.push_back(std::move(default_value));
	}

	void push_kwarg_default(std::shared_ptr<ASTNode> default_value)
	{
		m_kw_defaults.push_back(std::move(default_value));
	}

	void set_arg(std::shared_ptr<Argument> arg) { m_vararg = std::move(arg); }
	void set_kwarg(std::shared_ptr<Argument> arg) { m_kwarg = std::move(arg); }

	const std::vector<std::shared_ptr<Argument>> &posonlyargs() const { return m_posonlyargs; }
	const std::vector<std::shared_ptr<Argument>> &args() const { return m_args; }
	const std::shared_ptr<Argument> &vararg() const { return m_vararg; }
	const std::vector<std::shared_ptr<Argument>> &kwonlyargs() const { return m_kwonlyargs; }
	const std::vector<std::shared_ptr<ASTNode>> &kw_defaults() const { return m_kw_defaults; }
	const std::shared_ptr<Argument> &kwarg() const { return m_kwarg; }
	const std::vector<std::shared_ptr<ASTNode>> &defaults() const { return m_defaults; }

	Value *codegen(CodeGenerator *) const override;
};

class FunctionDefinition final : public ASTNode
{
	const std::string m_function_name;
	const std::shared_ptr<Arguments> m_args;
	std::vector<std::shared_ptr<ASTNode>> m_body;
	std::vector<std::shared_ptr<ASTNode>> m_decorator_list;
	const std::shared_ptr<ASTNode> m_returns;
	std::string m_type_comment;

	void print_this_node(const std::string &indent) const final;

  public:
	FunctionDefinition(std::string function_name,
		std::shared_ptr<Arguments> args,
		std::vector<std::shared_ptr<ASTNode>> body,
		std::vector<std::shared_ptr<ASTNode>> decorator_list,
		std::shared_ptr<ASTNode> returns,
		std::string type_comment,
		SourceLocation location)
		: ASTNode(ASTNodeType::FunctionDefinition, location),
		  m_function_name(std::move(function_name)), m_args(std::move(args)),
		  m_body(std::move(body)), m_decorator_list(std::move(decorator_list)),
		  m_returns(std::move(returns)), m_type_comment(std::move(type_comment))
	{}

	const std::string &name() const { return m_function_name; }
	const std::shared_ptr<Arguments> &args() const { return m_args; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	std::vector<std::shared_ptr<ASTNode>> &body() { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &decorator_list() const { return m_decorator_list; }
	const std::shared_ptr<ASTNode> &returns() const { return m_returns; }
	const std::string &type_comment() const { return m_type_comment; }

	void add_decorator(std::shared_ptr<ASTNode> decorator)
	{
		m_decorator_list.push_back(std::move(decorator));
	}

	Value *codegen(CodeGenerator *) const override;
};

class AsyncFunctionDefinition final : public ASTNode
{
	const std::string m_function_name;
	const std::shared_ptr<Arguments> m_args;
	std::vector<std::shared_ptr<ASTNode>> m_body;
	std::vector<std::shared_ptr<ASTNode>> m_decorator_list;
	const std::shared_ptr<ASTNode> m_returns;
	std::string m_type_comment;

	void print_this_node(const std::string &indent) const final;

  public:
	AsyncFunctionDefinition(std::string function_name,
		std::shared_ptr<Arguments> args,
		std::vector<std::shared_ptr<ASTNode>> body,
		std::vector<std::shared_ptr<ASTNode>> decorator_list,
		std::shared_ptr<ASTNode> returns,
		std::string type_comment,
		SourceLocation location)
		: ASTNode(ASTNodeType::AsyncFunctionDefinition, location),
		  m_function_name(std::move(function_name)), m_args(std::move(args)),
		  m_body(std::move(body)), m_decorator_list(std::move(decorator_list)),
		  m_returns(std::move(returns)), m_type_comment(std::move(type_comment))
	{}

	const std::string &name() const { return m_function_name; }
	const std::shared_ptr<Arguments> &args() const { return m_args; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	std::vector<std::shared_ptr<ASTNode>> &body() { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &decorator_list() const { return m_decorator_list; }
	const std::shared_ptr<ASTNode> &returns() const { return m_returns; }
	const std::string &type_comment() const { return m_type_comment; }

	void add_decorator(std::shared_ptr<ASTNode> decorator)
	{
		m_decorator_list.push_back(std::move(decorator));
	}

	Value *codegen(CodeGenerator *) const override;
};

class Lambda final : public ASTNode
{
	const std::shared_ptr<Arguments> m_args;
	std::shared_ptr<ASTNode> m_body;

	void print_this_node(const std::string &indent) const final;

  public:
	Lambda(std::shared_ptr<Arguments> args, std::shared_ptr<ASTNode> body, SourceLocation location)
		: ASTNode(ASTNodeType::Lambda, location), m_args(std::move(args)), m_body(std::move(body))
	{}

	const std::shared_ptr<Arguments> &args() const { return m_args; }
	const std::shared_ptr<ASTNode> &body() const { return m_body; }
	std::shared_ptr<ASTNode> &body() { return m_body; }

	Value *codegen(CodeGenerator *) const override;
};


class Keyword : public ASTNode
{
	std::optional<std::string> m_arg;
	std::shared_ptr<ASTNode> m_value;

  public:
	Keyword(std::shared_ptr<ASTNode> value, SourceLocation source_location)
		: ASTNode(ASTNodeType::Keyword, source_location), m_value(std::move(value))
	{}

	Keyword(std::string arg, std::shared_ptr<ASTNode> value, SourceLocation source_location)
		: ASTNode(ASTNodeType::Keyword, source_location), m_arg(std::move(arg)),
		  m_value(std::move(value))
	{}

	void print_this_node(const std::string &indent) const final;

	const std::optional<std::string> &arg() const { return m_arg; }
	std::shared_ptr<ASTNode> value() const { return m_value; }

	Value *codegen(CodeGenerator *) const override;
};


class ClassDefinition final : public ASTNode
{
	const std::string m_class_name;
	const std::vector<std::shared_ptr<ASTNode>> m_bases;
	const std::vector<std::shared_ptr<Keyword>> m_keywords;
	std::vector<std::shared_ptr<ASTNode>> m_body;
	std::vector<std::shared_ptr<ASTNode>> m_decorator_list;

	void print_this_node(const std::string &indent) const final;

  public:
	ClassDefinition(std::string class_name,
		std::vector<std::shared_ptr<ASTNode>> bases,
		std::vector<std::shared_ptr<Keyword>> keywords,
		std::vector<std::shared_ptr<ASTNode>> body,
		std::vector<std::shared_ptr<ASTNode>> decorator_list,
		SourceLocation location)
		: ASTNode(ASTNodeType::ClassDefinition, location), m_class_name(std::move(class_name)),
		  m_bases(std::move(bases)), m_keywords(std::move(keywords)), m_body(std::move(body)),
		  m_decorator_list(std::move(decorator_list))
	{}

	const std::string &name() const { return m_class_name; }
	const std::vector<std::shared_ptr<ASTNode>> &bases() const { return m_bases; }
	const std::vector<std::shared_ptr<Keyword>> &keywords() const { return m_keywords; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	std::vector<std::shared_ptr<ASTNode>> &body() { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &decorator_list() const { return m_decorator_list; }

	void add_decorator(std::shared_ptr<ASTNode> decorator)
	{
		m_decorator_list.push_back(std::move(decorator));
	}

	Value *codegen(CodeGenerator *) const override;
};


class Call : public ASTNode
{
	std::shared_ptr<ASTNode> m_function;
	std::vector<std::shared_ptr<ASTNode>> m_args;
	std::vector<std::shared_ptr<Keyword>> m_keywords;

	void print_this_node(const std::string &indent) const final;

  public:
	Call(std::shared_ptr<ASTNode> function,
		std::vector<std::shared_ptr<ASTNode>> args,
		std::vector<std::shared_ptr<Keyword>> keywords,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Call, source_location), m_function(std::move(function)),
		  m_args(std::move(args)), m_keywords(std::move(keywords))
	{}

	Call(std::shared_ptr<ASTNode> function, SourceLocation source_location)
		: Call(function, {}, {}, source_location)
	{}

	const std::shared_ptr<ASTNode> &function() const { return m_function; }
	const std::vector<std::shared_ptr<ASTNode>> &args() const { return m_args; }
	const std::vector<std::shared_ptr<Keyword>> &keywords() const { return m_keywords; }

	Value *codegen(CodeGenerator *) const override;
};

class Module : public ASTNode
{
	std::string m_filename;
	std::vector<std::shared_ptr<ASTNode>> m_body;

  public:
	Module(std::string filename)
		: ASTNode(ASTNodeType::Module, SourceLocation{}), m_filename(std::move(filename))
	{}

	template<typename T> void emplace(T node) { m_body.emplace_back(std::move(node)); }

	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	std::vector<std::shared_ptr<ASTNode>> &body() { return m_body; }

	const std::string &filename() const { return m_filename; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class If : public ASTNode
{
	std::shared_ptr<ASTNode> m_test;
	std::vector<std::shared_ptr<ASTNode>> m_body;
	std::vector<std::shared_ptr<ASTNode>> m_orelse;

  public:
	If(std::shared_ptr<ASTNode> test,
		std::vector<std::shared_ptr<ASTNode>> body,
		std::vector<std::shared_ptr<ASTNode>> orelse,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::If, source_location), m_test(std::move(test)),
		  m_body(std::move(body)), m_orelse(std::move(orelse))
	{}

	const std::shared_ptr<ASTNode> &test() const { return m_test; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &orelse() const { return m_orelse; }
	std::vector<std::shared_ptr<ASTNode>> &body() { return m_body; }
	std::vector<std::shared_ptr<ASTNode>> &orelse() { return m_orelse; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class For : public ASTNode
{
	std::shared_ptr<ASTNode> m_target;
	std::shared_ptr<ASTNode> m_iter;
	std::vector<std::shared_ptr<ASTNode>> m_body;
	std::vector<std::shared_ptr<ASTNode>> m_orelse;
	std::string m_type_comment;

  public:
	For(std::shared_ptr<ASTNode> target,
		std::shared_ptr<ASTNode> iter,
		std::vector<std::shared_ptr<ASTNode>> body,
		std::vector<std::shared_ptr<ASTNode>> orelse,
		std::string type_comment,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::For, source_location), m_target(std::move(target)),
		  m_iter(std::move(iter)), m_body(std::move(body)), m_orelse(std::move(orelse)),
		  m_type_comment(type_comment)
	{}

	const std::shared_ptr<ASTNode> &target() const { return m_target; }
	const std::shared_ptr<ASTNode> &iter() const { return m_iter; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &orelse() const { return m_orelse; }
	std::vector<std::shared_ptr<ASTNode>> &body() { return m_body; }
	std::vector<std::shared_ptr<ASTNode>> &orelse() { return m_orelse; }
	const std::string &type_comment() const { return m_type_comment; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class While : public ASTNode
{
	std::shared_ptr<ASTNode> m_test;
	std::vector<std::shared_ptr<ASTNode>> m_body;
	std::vector<std::shared_ptr<ASTNode>> m_orelse;

  public:
	While(std::shared_ptr<ASTNode> test,
		std::vector<std::shared_ptr<ASTNode>> body,
		std::vector<std::shared_ptr<ASTNode>> orelse,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::While, source_location), m_test(std::move(test)),
		  m_body(std::move(body)), m_orelse(std::move(orelse))
	{}

	const std::shared_ptr<ASTNode> &test() const { return m_test; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &orelse() const { return m_orelse; }
	std::vector<std::shared_ptr<ASTNode>> &body() { return m_body; }
	std::vector<std::shared_ptr<ASTNode>> &orelse() { return m_orelse; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


#define COMPARE_OPERATIONS \
	__COMPARE_OP(Eq)       \
	__COMPARE_OP(NotEq)    \
	__COMPARE_OP(Lt)       \
	__COMPARE_OP(LtE)      \
	__COMPARE_OP(Gt)       \
	__COMPARE_OP(GtE)      \
	__COMPARE_OP(Is)       \
	__COMPARE_OP(IsNot)    \
	__COMPARE_OP(In)       \
	__COMPARE_OP(NotIn)

class Compare : public ASTNode
{
  public:
	enum class OpType {
#define __COMPARE_OP(x) x,
		COMPARE_OPERATIONS
#undef __COMPARE_OP
	};

  private:
	std::shared_ptr<ASTNode> m_lhs;
	std::vector<OpType> m_ops;
	std::vector<std::shared_ptr<ASTNode>> m_comparators;

  public:
	Compare(std::shared_ptr<ASTNode> lhs,
		std::vector<OpType> &&ops,
		std::vector<std::shared_ptr<ASTNode>> &&comparators,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Compare, source_location), m_lhs(std::move(lhs)),
		  m_ops(std::move(ops)), m_comparators(std::move(comparators))
	{}

	const std::shared_ptr<ASTNode> &lhs() const { return m_lhs; }
	std::vector<OpType> ops() const { return m_ops; }
	const std::vector<std::shared_ptr<ASTNode>> &comparators() const { return m_comparators; }
	std::vector<std::shared_ptr<ASTNode>> &comparators() { return m_comparators; }

	Value *codegen(CodeGenerator *) const override;

  private:
	std::string_view op_type_to_string(OpType type) const
	{
		switch (type) {
#define __COMPARE_OP(x) \
	case OpType::x:     \
		return #x;
			COMPARE_OPERATIONS
#undef __COMPARE_OP
		}
		ASSERT_NOT_REACHED();
	}

	void print_this_node(const std::string &indent) const override;
};


class Attribute : public ASTNode
{
	std::shared_ptr<ASTNode> m_value;
	std::string m_attr;
	ContextType m_ctx;

  public:
	Attribute(std::shared_ptr<ASTNode> value,
		std::string attr,
		ContextType ctx,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Attribute, source_location), m_value(std::move(value)),
		  m_attr(std::move(attr)), m_ctx(ctx)
	{}

	const std::shared_ptr<ASTNode> &value() const { return m_value; }
	const std::string &attr() const { return m_attr; }
	ContextType context() const { return m_ctx; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

struct alias
{
	std::string name;
	std::string asname;
};

class ImportBase : public ASTNode
{
  protected:
	std::vector<alias> m_names;

  public:
	ImportBase(ASTNodeType node_type, std::vector<alias> &&names, SourceLocation source_location)
		: ASTNode(node_type, source_location), m_names(std::move(names))
	{}

	const std::vector<alias> &names() const { return m_names; }
};

class Import : public ImportBase
{
  public:
	Import(std::vector<alias> &&names, SourceLocation source_location)
		: ImportBase(ASTNodeType::Import, std::move(names), source_location)
	{}

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class ImportFrom : public ImportBase
{
	std::string m_module;
	size_t m_level{ 0 };

  public:
	ImportFrom(std::string module,
		std::vector<alias> &&names,
		size_t level,
		SourceLocation source_location)
		: ImportBase(ASTNodeType::ImportFrom, std::move(names), source_location),
		  m_module(std::move(module)), m_level(level)
	{}

	const std::string &module() const { return m_module; }
	size_t level() const { return m_level; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class Subscript : public ASTNode
{
  public:
	struct Index
	{
		std::shared_ptr<ASTNode> value;

		void print(const std::string &indent) const;
	};

	struct Slice
	{
		std::shared_ptr<ASTNode> lower;
		std::shared_ptr<ASTNode> upper;
		std::shared_ptr<ASTNode> step{ nullptr };
		void print(const std::string &indent) const;
	};

	struct ExtSlice
	{
		std::vector<std::variant<Index, Slice>> dims;
		void print(const std::string &indent) const;
	};

	using SliceType = std::variant<Index, Slice, ExtSlice>;

  private:
	std::shared_ptr<ASTNode> m_value;
	std::optional<SliceType> m_slice;
	ContextType m_ctx;

  public:
	Subscript(SourceLocation source_location) : ASTNode(ASTNodeType::Subscript, source_location) {}

	Subscript(std::shared_ptr<ASTNode> value,
		SliceType slice,
		ContextType ctx,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Subscript, source_location), m_value(std::move(value)),
		  m_slice(std::move(slice)), m_ctx(ctx)
	{}

	const std::shared_ptr<ASTNode> &value() const { return m_value; }
	const SliceType &slice() const
	{
		ASSERT(m_slice)
		return *m_slice;
	}
	ContextType context() const { return m_ctx; }

	void set_value(std::shared_ptr<ASTNode> value) { m_value = std::move(value); }
	void set_slice(SliceType slice) { m_slice = std::move(slice); }
	void set_context(ContextType context) { m_ctx = context; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class Raise : public ASTNode
{
  public:
	std::shared_ptr<ASTNode> m_exception;
	std::shared_ptr<ASTNode> m_cause;

  public:
	Raise(SourceLocation source_location) : ASTNode(ASTNodeType::Raise, source_location) {}

	Raise(std::shared_ptr<ASTNode> exception,
		std::shared_ptr<ASTNode> cause,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Raise, source_location), m_exception(std::move(exception)),
		  m_cause(std::move(cause))
	{}

	const std::shared_ptr<ASTNode> &exception() const { return m_exception; }
	const std::shared_ptr<ASTNode> &cause() const { return m_cause; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class ExceptHandler : public ASTNode
{
  public:
	std::shared_ptr<ASTNode> m_type;
	const std::string m_name;
	std::vector<std::shared_ptr<ASTNode>> m_body;

  public:
	ExceptHandler(std::shared_ptr<ASTNode> type,
		std::string name,
		std::vector<std::shared_ptr<ASTNode>> body,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::ExceptHandler, source_location), m_type(std::move(type)),
		  m_name(std::move(name)), m_body(std::move(body))
	{}

	const std::shared_ptr<ASTNode> &type() const { return m_type; }
	const std::string &name() const { return m_name; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	std::vector<std::shared_ptr<ASTNode>> &body() { return m_body; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class Try : public ASTNode
{
  public:
	std::vector<std::shared_ptr<ASTNode>> m_body;
	std::vector<std::shared_ptr<ExceptHandler>> m_handlers;
	std::vector<std::shared_ptr<ASTNode>> m_orelse;
	std::vector<std::shared_ptr<ASTNode>> m_finalbody;

  public:
	Try(std::vector<std::shared_ptr<ASTNode>> body,
		std::vector<std::shared_ptr<ExceptHandler>> handlers,
		std::vector<std::shared_ptr<ASTNode>> orelse,
		std::vector<std::shared_ptr<ASTNode>> finalbody,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Try, source_location), m_body(std::move(body)),
		  m_handlers(std::move(handlers)), m_orelse(std::move(orelse)),
		  m_finalbody(std::move(finalbody))
	{}

	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	std::vector<std::shared_ptr<ASTNode>> &body() { return m_body; }
	const std::vector<std::shared_ptr<ExceptHandler>> &handlers() const { return m_handlers; }
	const std::vector<std::shared_ptr<ASTNode>> &orelse() const { return m_orelse; }
	const std::vector<std::shared_ptr<ASTNode>> &finalbody() const { return m_finalbody; }
	std::vector<std::shared_ptr<ASTNode>> &orelse() { return m_orelse; }
	std::vector<std::shared_ptr<ASTNode>> &finalbody() { return m_finalbody; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class Assert : public ASTNode
{
  public:
	std::shared_ptr<ASTNode> m_test{ nullptr };
	std::shared_ptr<ASTNode> m_msg{ nullptr };

  public:
	Assert(std::shared_ptr<ASTNode> test,
		std::shared_ptr<ASTNode> msg,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Assert, source_location), m_test(std::move(test)),
		  m_msg(std::move(msg))
	{
		ASSERT(m_test)
	}

	const std::shared_ptr<ASTNode> &test() const { return m_test; }
	const std::shared_ptr<ASTNode> &msg() const { return m_msg; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


#define BOOL_OPERATIONS \
	__COMPARE_OP(And)   \
	__COMPARE_OP(Or)

class BoolOp : public ASTNode
{
  public:
	enum class OpType {
#define __COMPARE_OP(x) x,
		BOOL_OPERATIONS
#undef __COMPARE_OP
	};

  private:
	OpType m_op;
	std::vector<std::shared_ptr<ASTNode>> m_values;

  public:
	BoolOp(OpType op, std::vector<std::shared_ptr<ASTNode>> values, SourceLocation source_location)
		: ASTNode(ASTNodeType::BoolOp, source_location), m_op(op), m_values(std::move(values))
	{
		ASSERT(m_values.size() >= 2);
	}

	OpType op() const { return m_op; }
	const std::vector<std::shared_ptr<ASTNode>> &values() const { return m_values; }

	Value *codegen(CodeGenerator *) const override;

  private:
	std::string_view op_type_to_string(OpType type) const
	{
		switch (type) {
#define __COMPARE_OP(x) \
	case OpType::x:     \
		return #x;
			BOOL_OPERATIONS
#undef __COMPARE_OP
		}
		ASSERT_NOT_REACHED();
	}

	void print_this_node(const std::string &indent) const override;
};

class Pass : public ASTNode
{
  public:
	Pass(SourceLocation source_location) : ASTNode(ASTNodeType::Pass, source_location) {}

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class Continue : public ASTNode
{
  public:
	Continue(SourceLocation source_location) : ASTNode(ASTNodeType::Continue, source_location) {}

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class Break : public ASTNode
{
  public:
	Break(SourceLocation source_location) : ASTNode(ASTNodeType::Break, source_location) {}

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class Global : public ASTNode
{
	std::vector<std::string> m_names;

  public:
	Global(std::vector<std::string> names, SourceLocation source_location)
		: ASTNode(ASTNodeType::Global, source_location), m_names(std::move(names))
	{}

	const std::vector<std::string> &names() const { return m_names; }
	void add_name(const std::string &name) { m_names.push_back(name); }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class NonLocal : public ASTNode
{
	std::vector<std::string> m_names;

  public:
	NonLocal(std::vector<std::string> names, SourceLocation source_location)
		: ASTNode(ASTNodeType::NonLocal, source_location), m_names(std::move(names))
	{}

	const std::vector<std::string> &names() const { return m_names; }
	void add_name(const std::string &name) { m_names.push_back(name); }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class Delete : public ASTNode
{
	std::vector<std::shared_ptr<ASTNode>> m_targets;

  public:
	Delete(std::vector<std::shared_ptr<ASTNode>> targets, SourceLocation source_location)
		: ASTNode(ASTNodeType::Delete, source_location), m_targets(std::move(targets))
	{}

	const std::vector<std::shared_ptr<ASTNode>> &targets() const { return m_targets; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class WithItem : public ASTNode
{
	std::shared_ptr<ASTNode> m_context_expr;
	std::shared_ptr<ASTNode> m_optional_vars;

  public:
	WithItem(std::shared_ptr<ASTNode> context_expr,
		std::shared_ptr<ASTNode> optional_vars,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::WithItem, source_location), m_context_expr(std::move(context_expr)),
		  m_optional_vars(std::move(optional_vars))
	{}

	const std::shared_ptr<ASTNode> &context_expr() const { return m_context_expr; }
	const std::shared_ptr<ASTNode> &optional_vars() const { return m_optional_vars; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class With : public ASTNode
{
	std::vector<std::shared_ptr<WithItem>> m_items;
	std::vector<std::shared_ptr<ASTNode>> m_body;
	const std::string m_type_comment;

  public:
	With(std::vector<std::shared_ptr<WithItem>> items,
		std::vector<std::shared_ptr<ASTNode>> body,
		std::string type_comment,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::With, source_location), m_items(std::move(items)),
		  m_body(std::move(body)), m_type_comment(std::move(type_comment))
	{}

	const std::vector<std::shared_ptr<WithItem>> &items() const { return m_items; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	std::vector<std::shared_ptr<ASTNode>> &body() { return m_body; }
	const std::string &type_comment() const { return m_type_comment; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class IfExpr : public ASTNode
{
	std::shared_ptr<ASTNode> m_test;
	std::shared_ptr<ASTNode> m_body;
	std::shared_ptr<ASTNode> m_orelse;

  public:
	IfExpr(std::shared_ptr<ASTNode> test,
		std::shared_ptr<ASTNode> body,
		std::shared_ptr<ASTNode> orelse,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::IfExpr, source_location), m_test(std::move(test)),
		  m_body(std::move(body)), m_orelse(std::move(orelse))
	{}

	const std::shared_ptr<ASTNode> &test() const { return m_test; }
	const std::shared_ptr<ASTNode> &body() const { return m_body; }
	const std::shared_ptr<ASTNode> &orelse() const { return m_orelse; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class Starred : public ASTNode
{
	std::shared_ptr<ASTNode> m_value;
	ContextType m_ctx;

  public:
	Starred(std::shared_ptr<ASTNode> value, ContextType ctx, SourceLocation source_location)
		: ASTNode(ASTNodeType::Starred, source_location), m_value(std::move(value)), m_ctx(ctx)
	{}

	const std::shared_ptr<ASTNode> &value() const { return m_value; }
	ContextType ctx() const { return m_ctx; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class NamedExpr : public ASTNode
{
	std::shared_ptr<ASTNode> m_target;
	std::shared_ptr<ASTNode> m_value;

  public:
	NamedExpr(std::shared_ptr<ASTNode> target,
		std::shared_ptr<ASTNode> value,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::NamedExpr, source_location), m_target(std::move(target)),
		  m_value(std::move(value))
	{}

	const std::shared_ptr<ASTNode> &target() const { return m_target; }
	const std::shared_ptr<ASTNode> &value() const { return m_value; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class Comprehension : public ASTNode
{
	std::shared_ptr<ASTNode> m_target;
	std::shared_ptr<ASTNode> m_iter;
	std::vector<std::shared_ptr<ASTNode>> m_ifs;
	const bool m_is_async;

  public:
	Comprehension(std::shared_ptr<ASTNode> target,
		std::shared_ptr<ASTNode> iter,
		std::vector<std::shared_ptr<ASTNode>> ifs,
		bool is_async,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Comprehension, source_location), m_target(target), m_iter(iter),
		  m_ifs(ifs), m_is_async(is_async)
	{}

	const std::shared_ptr<ASTNode> &target() const { return m_target; }
	const std::shared_ptr<ASTNode> &iter() const { return m_iter; }
	const std::vector<std::shared_ptr<ASTNode>> &ifs() const { return m_ifs; }
	std::vector<std::shared_ptr<ASTNode>> &ifs() { return m_ifs; }

	bool is_async() const { return m_is_async; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class ListComp : public ASTNode
{
	std::shared_ptr<ASTNode> m_elt;
	std::vector<std::shared_ptr<Comprehension>> m_generators;

  public:
	ListComp(std::shared_ptr<ASTNode> elt,
		std::vector<std::shared_ptr<Comprehension>> &&generators,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::ListComp, source_location), m_elt(std::move(elt)),
		  m_generators(std::move(generators))
	{}

	const std::shared_ptr<ASTNode> elt() const { return m_elt; }
	const std::vector<std::shared_ptr<Comprehension>> &generators() const { return m_generators; }
	std::vector<std::shared_ptr<Comprehension>> &generators() { return m_generators; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class DictComp : public ASTNode
{
	std::shared_ptr<ASTNode> m_key;
	std::shared_ptr<ASTNode> m_value;
	std::vector<std::shared_ptr<Comprehension>> m_generators;

  public:
	DictComp(std::shared_ptr<ASTNode> key,
		std::shared_ptr<ASTNode> value,
		std::vector<std::shared_ptr<Comprehension>> &&generators,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::DictComp, source_location), m_key(std::move(key)),
		  m_value(std::move(value)), m_generators(std::move(generators))
	{}

	const std::shared_ptr<ASTNode> key() const { return m_key; }
	const std::shared_ptr<ASTNode> value() const { return m_value; }
	const std::vector<std::shared_ptr<Comprehension>> &generators() const { return m_generators; }
	std::vector<std::shared_ptr<Comprehension>> &generators() { return m_generators; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class GeneratorExp : public ASTNode
{
	std::shared_ptr<ASTNode> m_elt;
	std::vector<std::shared_ptr<Comprehension>> m_generators;

  public:
	GeneratorExp(std::shared_ptr<ASTNode> elt,
		std::vector<std::shared_ptr<Comprehension>> &&generators,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::GeneratorExp, source_location), m_elt(std::move(elt)),
		  m_generators(std::move(generators))
	{}

	const std::shared_ptr<ASTNode> elt() const { return m_elt; }
	const std::vector<std::shared_ptr<Comprehension>> &generators() const { return m_generators; }
	std::vector<std::shared_ptr<Comprehension>> &generators() { return m_generators; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class SetComp : public ASTNode
{
	std::shared_ptr<ASTNode> m_elt;
	std::vector<std::shared_ptr<Comprehension>> m_generators;

  public:
	SetComp(std::shared_ptr<ASTNode> elt,
		std::vector<std::shared_ptr<Comprehension>> &&generators,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::SetComp, source_location), m_elt(std::move(elt)),
		  m_generators(std::move(generators))
	{}

	const std::shared_ptr<ASTNode> elt() const { return m_elt; }
	const std::vector<std::shared_ptr<Comprehension>> &generators() const { return m_generators; }
	std::vector<std::shared_ptr<Comprehension>> &generators() { return m_generators; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class JoinedStr : public ASTNode
{
	std::vector<std::shared_ptr<ASTNode>> m_values;

  public:
	JoinedStr(std::vector<std::shared_ptr<ASTNode>> values, SourceLocation source_location)
		: ASTNode(ASTNodeType::JoinedStr, source_location), m_values(std::move(values))
	{}

	const std::vector<std::shared_ptr<ASTNode>> &values() const { return m_values; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class FormattedValue : public ASTNode
{
  public:
	enum class Conversion { NONE, STRING, REPR, ASCII };

  private:
	std::shared_ptr<ASTNode> m_value;
	Conversion m_conversion;
	std::shared_ptr<JoinedStr> m_format_spec;

  public:
	FormattedValue(std::shared_ptr<ASTNode> value,
		Conversion conversion,
		std::shared_ptr<JoinedStr> format_spec,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::FormattedValue, source_location), m_value(std::move(value)),
		  m_conversion(conversion), m_format_spec(std::move(format_spec))
	{}

	const std::shared_ptr<ASTNode> &value() const { return m_value; }
	Conversion conversion() const { return m_conversion; }
	const std::shared_ptr<JoinedStr> &format_spec() const { return m_format_spec; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


template<typename NodeType> std::shared_ptr<NodeType> as(std::shared_ptr<ASTNode> node);

#define __AST_NODE_TYPE(x) template<> std::shared_ptr<x> as(std::shared_ptr<ASTNode> node);
AST_NODE_TYPES
#undef __AST_NODE_TYPE

struct CodeGenerator
{
	virtual ~CodeGenerator() = default;
	std::vector<std::unique_ptr<Value>> m_values;
#define __AST_NODE_TYPE(NodeType) virtual Value *visit(const NodeType *node) = 0;
	AST_NODE_TYPES
#undef __AST_NODE_TYPE
};

struct NodeVisitor
{
	virtual ~NodeVisitor() = default;

#define __AST_NODE_TYPE(NodeType) virtual void visit(NodeType *node);
	AST_NODE_TYPES
#undef __AST_NODE_TYPE

  protected:
	void dispatch(ASTNode *node);
#define __AST_NODE_TYPE(NodeType) void dispatch(NodeType *node);
	AST_NODE_TYPES
#undef __AST_NODE_TYPE
};

struct NodeTransformVisitor
{
	virtual ~NodeTransformVisitor() = default;

	bool m_can_return_multiple_nodes{ false };
#define __AST_NODE_TYPE(NodeType) \
	virtual std::vector<std::shared_ptr<ASTNode>> visit(std::shared_ptr<NodeType> node);
	AST_NODE_TYPES
#undef __AST_NODE_TYPE

  protected:
	void transform_single_node(std::shared_ptr<ASTNode> node);

	void transform_multiple_nodes(std::vector<std::shared_ptr<ASTNode>> &nodes);
};


}// namespace ast
