#pragma once

#include <memory>
#include <numeric>
#include <optional>
#include <stack>
#include <string>
#include <variant>
#include <vector>

#include <gmpxx.h>

#include "ast/ASTArena.hpp"
#include "forward.hpp"
#include "lexer/Lexer.hpp"
#include "utilities.hpp"

#include "spdlog/spdlog.h"

struct SourceLocation
{
	Position start;
	Position end;
	auto operator<=>(const SourceLocation &other) const = default;
	friend std::ostream &operator<<(std::ostream &os, const SourceLocation &sc)
	{
		os << '[' << sc.start << '-' << sc.end << ']';
		return os;
	}
};

template<> struct fmt::formatter<SourceLocation>
{
	constexpr auto parse(format_parse_context &ctx) { return ctx.end(); }

	template<typename FormatContext> auto format(const SourceLocation &sc, FormatContext &ctx)
	{
		return format_to(ctx.out(), "[{}-{}]", sc.start, sc.end);
	}
};


namespace ast {

#define AST_NODE_TYPES                       \
	__AST_NODE_TYPE(Argument)                \
	__AST_NODE_TYPE(Arguments)               \
	__AST_NODE_TYPE(Attribute)               \
	__AST_NODE_TYPE(Assign)                  \
	__AST_NODE_TYPE(Assert)                  \
	__AST_NODE_TYPE(AsyncFunctionDefinition) \
	__AST_NODE_TYPE(Await)                   \
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
	std::stack<const Arguments *> m_local_args;
	std::vector<const ASTNode *> m_parent_nodes;

  public:
	void push_local_args(const Arguments *args) { m_local_args.push(args); }
	void pop_local_args() { m_local_args.pop(); }

	bool has_local_args() const { return !m_local_args.empty(); }

	void push_node(const ASTNode *node) { m_parent_nodes.push_back(node); }
	void pop_node() { m_parent_nodes.pop_back(); }

	const Arguments *local_args() const { return m_local_args.top(); }
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
	ASTNode *m_value{ nullptr };

  public:
	Expression(ASTNode *value, SourceLocation source_location)
		: ASTNode(ASTNodeType::Expression, source_location), m_value(std::move(value))
	{}

	ASTNode *value() const { return m_value; }

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
	std::vector<ASTNode *> m_elements;
	ContextType m_ctx;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	List(std::vector<ASTNode *> elements, ContextType ctx, SourceLocation source_location)
		: ASTNode(ASTNodeType::List, source_location), m_elements(std::move(elements)), m_ctx(ctx)
	{}

	List(ContextType ctx, SourceLocation source_location)
		: ASTNode(ASTNodeType::List, source_location), m_elements(), m_ctx(ctx)
	{}

	void append(ASTNode *element) { m_elements.push_back(std::move(element)); }

	ContextType context() const { return m_ctx; }
	const std::vector<ASTNode *> &elements() const { return m_elements; }

	Value *codegen(CodeGenerator *) const override;
};

class Tuple : public ASTNode
{
  private:
	std::vector<ASTNode *> m_elements;
	ContextType m_ctx;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Tuple(std::vector<ASTNode *> elements, ContextType ctx, SourceLocation source_location)
		: ASTNode(ASTNodeType::Tuple, source_location), m_elements(std::move(elements)), m_ctx(ctx)
	{}

	Tuple(ContextType ctx, SourceLocation source_location)
		: ASTNode(ASTNodeType::Tuple, source_location), m_elements(), m_ctx(ctx)
	{}

	void append(ASTNode *element) { m_elements.push_back(std::move(element)); }

	ContextType context() const { return m_ctx; }
	const std::vector<ASTNode *> &elements() const { return m_elements; }
	std::vector<ASTNode *> &elements() { return m_elements; }

	Value *codegen(CodeGenerator *) const override;
};


class Dict : public ASTNode
{
  private:
	std::vector<ASTNode *> m_keys;
	std::vector<ASTNode *> m_values;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Dict(std::vector<ASTNode *> keys, std::vector<ASTNode *> values, SourceLocation source_location)
		: ASTNode(ASTNodeType::Dict, source_location), m_keys(std::move(keys)),
		  m_values(std::move(values))
	{}

	Dict(SourceLocation source_location)
		: ASTNode(ASTNodeType::Dict, source_location), m_keys(), m_values()
	{}

	const std::vector<ASTNode *> &keys() const { return m_keys; }
	const std::vector<ASTNode *> &values() const { return m_values; }

	Value *codegen(CodeGenerator *) const override;
};


class Set : public ASTNode
{
  private:
	std::vector<ASTNode *> m_elements;
	ContextType m_ctx;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Set(std::vector<ASTNode *> elements, ContextType ctx, SourceLocation source_location)
		: ASTNode(ASTNodeType::List, source_location), m_elements(std::move(elements)), m_ctx(ctx)
	{}

	ContextType context() const { return m_ctx; }
	const std::vector<ASTNode *> &elements() const { return m_elements; }

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
	std::vector<ASTNode *> m_targets;
	ASTNode *m_value{ nullptr };
	std::string m_type_comment;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Assign(std::vector<ASTNode *> targets,
		ASTNode *value,
		std::string type_comment,
		SourceLocation source_location)
		: Statement(ASTNodeType::Assign, source_location), m_targets(std::move(targets)),
		  m_value(std::move(value)), m_type_comment(std::move(type_comment))
	{}

	const std::vector<ASTNode *> &targets() const { return m_targets; }
	ASTNode *value() const { return m_value; }
	void set_value(ASTNode *v) { m_value = std::move(v); }

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
	ASTNode *m_operand{ nullptr };

  public:
	UnaryExpr(UnaryOpType op_type, ASTNode *operand, SourceLocation source_location)
		: ASTNode(ASTNodeType::UnaryExpr, source_location), m_op_type(op_type),
		  m_operand(std::move(operand))
	{}

	ASTNode *operand() const { return m_operand; }
	ASTNode *&operand() { return m_operand; }

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
	ASTNode *m_lhs{ nullptr };
	ASTNode *m_rhs{ nullptr };

  public:
	BinaryExpr(BinaryOpType op_type, ASTNode *lhs, ASTNode *rhs, SourceLocation source_location)
		: ASTNode(ASTNodeType::BinaryExpr, source_location), m_op_type(op_type),
		  m_lhs(std::move(lhs)), m_rhs(std::move(rhs))
	{}

	ASTNode *lhs() const { return m_lhs; }
	ASTNode *&lhs() { return m_lhs; }

	ASTNode *rhs() const { return m_rhs; }
	ASTNode *&rhs() { return m_rhs; }

	BinaryOpType op_type() const { return m_op_type; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class AugAssign : public Statement
{
	ASTNode *m_target{ nullptr };
	BinaryOpType m_op;
	ASTNode *m_value{ nullptr };

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	AugAssign(ASTNode *target, BinaryOpType op, ASTNode *value, SourceLocation source_location)
		: Statement(ASTNodeType::AugAssign, source_location), m_target(std::move(target)), m_op(op),
		  m_value(std::move(value))
	{}

	ASTNode *target() const { return m_target; }
	BinaryOpType op() const { return m_op; }
	ASTNode *value() const { return m_value; }
	void set_value(ASTNode *value) { m_value = std::move(value); }

	Value *codegen(CodeGenerator *) const override;
};

class Return : public ASTNode
{
	ASTNode *m_value{ nullptr };

  public:
	Return(ASTNode *value, SourceLocation source_location)
		: ASTNode(ASTNodeType::Return, source_location), m_value(std::move(value))
	{}

	ASTNode *value() const { return m_value; }

	void print_this_node(const std::string &indent) const override;

	Value *codegen(CodeGenerator *) const override;
};

class Yield : public ASTNode
{
	ASTNode *m_value{ nullptr };

  public:
	Yield(ASTNode *value, SourceLocation source_location)
		: ASTNode(ASTNodeType::Yield, source_location), m_value(std::move(value))
	{}

	ASTNode *value() const { return m_value; }

	void print_this_node(const std::string &indent) const override;

	Value *codegen(CodeGenerator *) const override;
};

class YieldFrom : public ASTNode
{
	ASTNode *m_value{ nullptr };

  public:
	YieldFrom(ASTNode *value, SourceLocation source_location)
		: ASTNode(ASTNodeType::YieldFrom, source_location), m_value(std::move(value))
	{}

	ASTNode *value() const { return m_value; }

	void print_this_node(const std::string &indent) const override;

	Value *codegen(CodeGenerator *) const override;
};

class Argument final : public ASTNode
{
	const std::string m_arg;
	ASTNode *m_annotation{ nullptr };
	const std::string m_type_comment;

  public:
	Argument(std::string arg,
		ASTNode *annotation,
		std::string type_comment,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Argument, source_location), m_arg(std::move(arg)),
		  m_annotation(std::move(annotation)), m_type_comment(std::move(type_comment))
	{}

	void print_this_node(const std::string &indent) const final;

	const std::string &name() const { return m_arg; }
	ASTNode *annotation() const { return m_annotation; }

	Value *codegen(CodeGenerator *) const override;
};


class Arguments : public ASTNode
{
	std::vector<Argument *> m_posonlyargs;
	std::vector<Argument *> m_args;
	Argument *m_vararg{ nullptr };
	std::vector<Argument *> m_kwonlyargs;
	std::vector<ASTNode *> m_kw_defaults;
	Argument *m_kwarg{ nullptr };
	std::vector<ASTNode *> m_defaults;

  public:
	Arguments(SourceLocation source_location) : ASTNode(ASTNodeType::Arguments, source_location) {}
	Arguments(std::vector<Argument *> args, SourceLocation source_location)
		: Arguments(source_location)
	{
		m_args = std::move(args);
	}

	Arguments(std::vector<Argument *> posonlyargs,
		std::vector<Argument *> args,
		Argument *vararg,
		std::vector<Argument *> kwonlyargs,
		std::vector<ASTNode *> kw_defaults,
		Argument *kwarg,
		std::vector<ASTNode *> defaults,
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

	void push_positional_arg(Argument *arg) { m_posonlyargs.push_back(std::move(arg)); }

	void push_arg(Argument *arg) { m_args.push_back(std::move(arg)); }

	std::vector<std::string> argument_names() const;

	std::vector<std::string> kw_only_argument_names() const;

	void push_kwonlyarg(Argument *kwarg) { m_kwonlyargs.push_back(std::move(kwarg)); }

	void push_default(ASTNode *default_value) { m_defaults.push_back(std::move(default_value)); }

	void push_kwarg_default(ASTNode *default_value)
	{
		m_kw_defaults.push_back(std::move(default_value));
	}

	void set_arg(Argument *arg) { m_vararg = std::move(arg); }
	void set_kwarg(Argument *arg) { m_kwarg = std::move(arg); }

	const std::vector<Argument *> &posonlyargs() const { return m_posonlyargs; }
	const std::vector<Argument *> &args() const { return m_args; }
	Argument *vararg() const { return m_vararg; }
	const std::vector<Argument *> &kwonlyargs() const { return m_kwonlyargs; }
	const std::vector<ASTNode *> &kw_defaults() const { return m_kw_defaults; }
	Argument *kwarg() const { return m_kwarg; }
	const std::vector<ASTNode *> &defaults() const { return m_defaults; }

	Value *codegen(CodeGenerator *) const override;
};

class FunctionDefinition final : public ASTNode
{
	const std::string m_function_name;
	Arguments *m_args{ nullptr };
	std::vector<ASTNode *> m_body;
	std::vector<ASTNode *> m_decorator_list;
	ASTNode *m_returns{ nullptr };
	std::string m_type_comment;

	void print_this_node(const std::string &indent) const final;

  public:
	FunctionDefinition(std::string function_name,
		Arguments *args,
		std::vector<ASTNode *> body,
		std::vector<ASTNode *> decorator_list,
		ASTNode *returns,
		std::string type_comment,
		SourceLocation location)
		: ASTNode(ASTNodeType::FunctionDefinition, location),
		  m_function_name(std::move(function_name)), m_args(std::move(args)),
		  m_body(std::move(body)), m_decorator_list(std::move(decorator_list)),
		  m_returns(std::move(returns)), m_type_comment(std::move(type_comment))
	{}

	const std::string &name() const { return m_function_name; }
	Arguments *args() const { return m_args; }
	const std::vector<ASTNode *> &body() const { return m_body; }
	std::vector<ASTNode *> &body() { return m_body; }
	const std::vector<ASTNode *> &decorator_list() const { return m_decorator_list; }
	ASTNode *returns() const { return m_returns; }
	const std::string &type_comment() const { return m_type_comment; }

	void add_decorator(ASTNode *decorator) { m_decorator_list.push_back(std::move(decorator)); }

	Value *codegen(CodeGenerator *) const override;
};

class AsyncFunctionDefinition final : public ASTNode
{
	const std::string m_function_name;
	Arguments *m_args{ nullptr };
	std::vector<ASTNode *> m_body;
	std::vector<ASTNode *> m_decorator_list;
	ASTNode *m_returns{ nullptr };
	std::string m_type_comment;

	void print_this_node(const std::string &indent) const final;

  public:
	AsyncFunctionDefinition(std::string function_name,
		Arguments *args,
		std::vector<ASTNode *> body,
		std::vector<ASTNode *> decorator_list,
		ASTNode *returns,
		std::string type_comment,
		SourceLocation location)
		: ASTNode(ASTNodeType::AsyncFunctionDefinition, location),
		  m_function_name(std::move(function_name)), m_args(std::move(args)),
		  m_body(std::move(body)), m_decorator_list(std::move(decorator_list)),
		  m_returns(std::move(returns)), m_type_comment(std::move(type_comment))
	{}

	const std::string &name() const { return m_function_name; }
	Arguments *args() const { return m_args; }
	const std::vector<ASTNode *> &body() const { return m_body; }
	std::vector<ASTNode *> &body() { return m_body; }
	const std::vector<ASTNode *> &decorator_list() const { return m_decorator_list; }
	ASTNode *returns() const { return m_returns; }
	const std::string &type_comment() const { return m_type_comment; }

	void add_decorator(ASTNode *decorator) { m_decorator_list.push_back(std::move(decorator)); }

	Value *codegen(CodeGenerator *) const override;
};

class Await final : public ASTNode
{
	ASTNode *m_value{ nullptr };

	void print_this_node(const std::string &indent) const final;

  public:
	Await(ASTNode *value, SourceLocation source_location)
		: ASTNode(ASTNodeType::Await, std::move(source_location)), m_value(std::move(value))
	{}

	ASTNode *value() const { return m_value; }

	Value *codegen(CodeGenerator *) const override;
};

class Lambda final : public ASTNode
{
	Arguments *m_args{ nullptr };
	ASTNode *m_body{ nullptr };

	void print_this_node(const std::string &indent) const final;

  public:
	Lambda(Arguments *args, ASTNode *body, SourceLocation location)
		: ASTNode(ASTNodeType::Lambda, location), m_args(std::move(args)), m_body(std::move(body))
	{}

	Arguments *args() const { return m_args; }
	ASTNode *body() const { return m_body; }
	ASTNode *&body() { return m_body; }

	Value *codegen(CodeGenerator *) const override;
};


class Keyword : public ASTNode
{
	std::optional<std::string> m_arg;
	ASTNode *m_value{ nullptr };

  public:
	Keyword(ASTNode *value, SourceLocation source_location)
		: ASTNode(ASTNodeType::Keyword, source_location), m_value(std::move(value))
	{}

	Keyword(std::string arg, ASTNode *value, SourceLocation source_location)
		: ASTNode(ASTNodeType::Keyword, source_location), m_arg(std::move(arg)),
		  m_value(std::move(value))
	{}

	void print_this_node(const std::string &indent) const final;

	const std::optional<std::string> &arg() const { return m_arg; }
	ASTNode *value() const { return m_value; }

	Value *codegen(CodeGenerator *) const override;
};


class ClassDefinition final : public ASTNode
{
	const std::string m_class_name;
	const std::vector<ASTNode *> m_bases;
	const std::vector<Keyword *> m_keywords;
	std::vector<ASTNode *> m_body;
	std::vector<ASTNode *> m_decorator_list;

	void print_this_node(const std::string &indent) const final;

  public:
	ClassDefinition(std::string class_name,
		std::vector<ASTNode *> bases,
		std::vector<Keyword *> keywords,
		std::vector<ASTNode *> body,
		std::vector<ASTNode *> decorator_list,
		SourceLocation location)
		: ASTNode(ASTNodeType::ClassDefinition, location), m_class_name(std::move(class_name)),
		  m_bases(std::move(bases)), m_keywords(std::move(keywords)), m_body(std::move(body)),
		  m_decorator_list(std::move(decorator_list))
	{}

	const std::string &name() const { return m_class_name; }
	const std::vector<ASTNode *> &bases() const { return m_bases; }
	const std::vector<Keyword *> &keywords() const { return m_keywords; }
	const std::vector<ASTNode *> &body() const { return m_body; }
	std::vector<ASTNode *> &body() { return m_body; }
	const std::vector<ASTNode *> &decorator_list() const { return m_decorator_list; }

	void add_decorator(ASTNode *decorator) { m_decorator_list.push_back(std::move(decorator)); }

	Value *codegen(CodeGenerator *) const override;
};


class Call : public ASTNode
{
	ASTNode *m_function{ nullptr };
	std::vector<ASTNode *> m_args;
	std::vector<Keyword *> m_keywords;

	void print_this_node(const std::string &indent) const final;

  public:
	Call(ASTNode *function,
		std::vector<ASTNode *> args,
		std::vector<Keyword *> keywords,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Call, source_location), m_function(std::move(function)),
		  m_args(std::move(args)), m_keywords(std::move(keywords))
	{}

	Call(ASTNode *function, SourceLocation source_location)
		: Call(function, {}, {}, source_location)
	{}

	ASTNode *function() const { return m_function; }
	const std::vector<ASTNode *> &args() const { return m_args; }
	const std::vector<Keyword *> &keywords() const { return m_keywords; }

	Value *codegen(CodeGenerator *) const override;
};

class Module : public ASTNode
{
	std::string m_filename;
	// The arena owns every child node transitively reachable from this Module.
	// Allocated nodes hold raw back-pointers; ownership lives solely in the arena.
	ASTArena m_arena;
	std::vector<ASTNode *> m_body;

  public:
	Module(std::string filename)
		: ASTNode(ASTNodeType::Module, SourceLocation{}), m_filename(std::move(filename))
	{}

	ASTArena &arena() { return m_arena; }
	const ASTArena &arena() const { return m_arena; }

	void emplace(ASTNode *node) { m_body.push_back(node); }

	const std::vector<ASTNode *> &body() const { return m_body; }
	std::vector<ASTNode *> &body() { return m_body; }

	const std::string &filename() const { return m_filename; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class If : public ASTNode
{
	ASTNode *m_test{ nullptr };
	std::vector<ASTNode *> m_body;
	std::vector<ASTNode *> m_orelse;

  public:
	If(ASTNode *test,
		std::vector<ASTNode *> body,
		std::vector<ASTNode *> orelse,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::If, source_location), m_test(std::move(test)),
		  m_body(std::move(body)), m_orelse(std::move(orelse))
	{}

	ASTNode *test() const { return m_test; }
	const std::vector<ASTNode *> &body() const { return m_body; }
	const std::vector<ASTNode *> &orelse() const { return m_orelse; }
	std::vector<ASTNode *> &body() { return m_body; }
	std::vector<ASTNode *> &orelse() { return m_orelse; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class For : public ASTNode
{
	ASTNode *m_target{ nullptr };
	ASTNode *m_iter{ nullptr };
	std::vector<ASTNode *> m_body;
	std::vector<ASTNode *> m_orelse;
	std::string m_type_comment;

  public:
	For(ASTNode *target,
		ASTNode *iter,
		std::vector<ASTNode *> body,
		std::vector<ASTNode *> orelse,
		std::string type_comment,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::For, source_location), m_target(std::move(target)),
		  m_iter(std::move(iter)), m_body(std::move(body)), m_orelse(std::move(orelse)),
		  m_type_comment(type_comment)
	{}

	ASTNode *target() const { return m_target; }
	ASTNode *iter() const { return m_iter; }
	const std::vector<ASTNode *> &body() const { return m_body; }
	const std::vector<ASTNode *> &orelse() const { return m_orelse; }
	std::vector<ASTNode *> &body() { return m_body; }
	std::vector<ASTNode *> &orelse() { return m_orelse; }
	const std::string &type_comment() const { return m_type_comment; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class While : public ASTNode
{
	ASTNode *m_test{ nullptr };
	std::vector<ASTNode *> m_body;
	std::vector<ASTNode *> m_orelse;

  public:
	While(ASTNode *test,
		std::vector<ASTNode *> body,
		std::vector<ASTNode *> orelse,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::While, source_location), m_test(std::move(test)),
		  m_body(std::move(body)), m_orelse(std::move(orelse))
	{}

	ASTNode *test() const { return m_test; }
	const std::vector<ASTNode *> &body() const { return m_body; }
	const std::vector<ASTNode *> &orelse() const { return m_orelse; }
	std::vector<ASTNode *> &body() { return m_body; }
	std::vector<ASTNode *> &orelse() { return m_orelse; }

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
	ASTNode *m_lhs{ nullptr };
	std::vector<OpType> m_ops;
	std::vector<ASTNode *> m_comparators;

  public:
	Compare(ASTNode *lhs,
		std::vector<OpType> &&ops,
		std::vector<ASTNode *> &&comparators,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Compare, source_location), m_lhs(std::move(lhs)),
		  m_ops(std::move(ops)), m_comparators(std::move(comparators))
	{}

	ASTNode *lhs() const { return m_lhs; }
	std::vector<OpType> ops() const { return m_ops; }
	const std::vector<ASTNode *> &comparators() const { return m_comparators; }
	std::vector<ASTNode *> &comparators() { return m_comparators; }

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
	ASTNode *m_value{ nullptr };
	std::string m_attr;
	ContextType m_ctx;

  public:
	Attribute(ASTNode *value, std::string attr, ContextType ctx, SourceLocation source_location)
		: ASTNode(ASTNodeType::Attribute, source_location), m_value(std::move(value)),
		  m_attr(std::move(attr)), m_ctx(ctx)
	{}

	ASTNode *value() const { return m_value; }
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
		ASTNode *value;

		void print(const std::string &indent) const;
	};

	struct Slice
	{
		ASTNode *lower;
		ASTNode *upper;
		ASTNode *step{ nullptr };
		void print(const std::string &indent) const;
	};

	struct ExtSlice
	{
		std::vector<std::variant<Index, Slice>> dims;
		void print(const std::string &indent) const;
	};

	using SliceType = std::variant<Index, Slice, ExtSlice>;

  private:
	ASTNode *m_value{ nullptr };
	std::optional<SliceType> m_slice;
	ContextType m_ctx;

  public:
	Subscript(SourceLocation source_location) : ASTNode(ASTNodeType::Subscript, source_location) {}

	Subscript(ASTNode *value, SliceType slice, ContextType ctx, SourceLocation source_location)
		: ASTNode(ASTNodeType::Subscript, source_location), m_value(std::move(value)),
		  m_slice(std::move(slice)), m_ctx(ctx)
	{}

	ASTNode *value() const { return m_value; }
	const SliceType &slice() const
	{
		ASSERT(m_slice);
		return *m_slice;
	}
	ContextType context() const { return m_ctx; }

	void set_value(ASTNode *value) { m_value = std::move(value); }
	void set_slice(SliceType slice) { m_slice = std::move(slice); }
	void set_context(ContextType context) { m_ctx = context; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class Raise : public ASTNode
{
  public:
	ASTNode *m_exception{ nullptr };
	ASTNode *m_cause{ nullptr };

  public:
	Raise(SourceLocation source_location) : ASTNode(ASTNodeType::Raise, source_location) {}

	Raise(ASTNode *exception, ASTNode *cause, SourceLocation source_location)
		: ASTNode(ASTNodeType::Raise, source_location), m_exception(std::move(exception)),
		  m_cause(std::move(cause))
	{}

	ASTNode *exception() const { return m_exception; }
	ASTNode *cause() const { return m_cause; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class ExceptHandler : public ASTNode
{
  public:
	ASTNode *m_type{ nullptr };
	const std::string m_name;
	std::vector<ASTNode *> m_body;

  public:
	ExceptHandler(ASTNode *type,
		std::string name,
		std::vector<ASTNode *> body,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::ExceptHandler, source_location), m_type(std::move(type)),
		  m_name(std::move(name)), m_body(std::move(body))
	{}

	ASTNode *type() const { return m_type; }
	const std::string &name() const { return m_name; }
	const std::vector<ASTNode *> &body() const { return m_body; }
	std::vector<ASTNode *> &body() { return m_body; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class Try : public ASTNode
{
  public:
	std::vector<ASTNode *> m_body;
	std::vector<ExceptHandler *> m_handlers;
	std::vector<ASTNode *> m_orelse;
	std::vector<ASTNode *> m_finalbody;

  public:
	Try(std::vector<ASTNode *> body,
		std::vector<ExceptHandler *> handlers,
		std::vector<ASTNode *> orelse,
		std::vector<ASTNode *> finalbody,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Try, source_location), m_body(std::move(body)),
		  m_handlers(std::move(handlers)), m_orelse(std::move(orelse)),
		  m_finalbody(std::move(finalbody))
	{}

	const std::vector<ASTNode *> &body() const { return m_body; }
	std::vector<ASTNode *> &body() { return m_body; }
	const std::vector<ExceptHandler *> &handlers() const { return m_handlers; }
	const std::vector<ASTNode *> &orelse() const { return m_orelse; }
	const std::vector<ASTNode *> &finalbody() const { return m_finalbody; }
	std::vector<ASTNode *> &orelse() { return m_orelse; }
	std::vector<ASTNode *> &finalbody() { return m_finalbody; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class Assert : public ASTNode
{
  public:
	ASTNode *m_test{ nullptr };
	ASTNode *m_msg{ nullptr };

  public:
	Assert(ASTNode *test, ASTNode *msg, SourceLocation source_location)
		: ASTNode(ASTNodeType::Assert, source_location), m_test(std::move(test)),
		  m_msg(std::move(msg))
	{
		ASSERT(m_test);
	}

	ASTNode *test() const { return m_test; }
	ASTNode *msg() const { return m_msg; }

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
	std::vector<ASTNode *> m_values;

  public:
	BoolOp(OpType op, std::vector<ASTNode *> values, SourceLocation source_location)
		: ASTNode(ASTNodeType::BoolOp, source_location), m_op(op), m_values(std::move(values))
	{
		ASSERT(m_values.size() >= 2);
	}

	OpType op() const { return m_op; }
	const std::vector<ASTNode *> &values() const { return m_values; }

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
	std::vector<ASTNode *> m_targets;

  public:
	Delete(std::vector<ASTNode *> targets, SourceLocation source_location)
		: ASTNode(ASTNodeType::Delete, source_location), m_targets(std::move(targets))
	{}

	const std::vector<ASTNode *> &targets() const { return m_targets; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class WithItem : public ASTNode
{
	ASTNode *m_context_expr{ nullptr };
	ASTNode *m_optional_vars{ nullptr };

  public:
	WithItem(ASTNode *context_expr, ASTNode *optional_vars, SourceLocation source_location)
		: ASTNode(ASTNodeType::WithItem, source_location), m_context_expr(std::move(context_expr)),
		  m_optional_vars(std::move(optional_vars))
	{}

	ASTNode *context_expr() const { return m_context_expr; }
	ASTNode *optional_vars() const { return m_optional_vars; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class With : public ASTNode
{
	std::vector<WithItem *> m_items;
	std::vector<ASTNode *> m_body;
	const std::string m_type_comment;

  public:
	With(std::vector<WithItem *> items,
		std::vector<ASTNode *> body,
		std::string type_comment,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::With, source_location), m_items(std::move(items)),
		  m_body(std::move(body)), m_type_comment(std::move(type_comment))
	{}

	const std::vector<WithItem *> &items() const { return m_items; }
	const std::vector<ASTNode *> &body() const { return m_body; }
	std::vector<ASTNode *> &body() { return m_body; }
	const std::string &type_comment() const { return m_type_comment; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class IfExpr : public ASTNode
{
	ASTNode *m_test{ nullptr };
	ASTNode *m_body{ nullptr };
	ASTNode *m_orelse{ nullptr };

  public:
	IfExpr(ASTNode *test, ASTNode *body, ASTNode *orelse, SourceLocation source_location)
		: ASTNode(ASTNodeType::IfExpr, source_location), m_test(std::move(test)),
		  m_body(std::move(body)), m_orelse(std::move(orelse))
	{}

	ASTNode *test() const { return m_test; }
	ASTNode *body() const { return m_body; }
	ASTNode *orelse() const { return m_orelse; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class Starred : public ASTNode
{
	ASTNode *m_value{ nullptr };
	ContextType m_ctx;

  public:
	Starred(ASTNode *value, ContextType ctx, SourceLocation source_location)
		: ASTNode(ASTNodeType::Starred, source_location), m_value(std::move(value)), m_ctx(ctx)
	{}

	ASTNode *value() const { return m_value; }
	ContextType ctx() const { return m_ctx; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class NamedExpr : public ASTNode
{
	ASTNode *m_target{ nullptr };
	ASTNode *m_value{ nullptr };

  public:
	NamedExpr(ASTNode *target, ASTNode *value, SourceLocation source_location)
		: ASTNode(ASTNodeType::NamedExpr, source_location), m_target(std::move(target)),
		  m_value(std::move(value))
	{}

	ASTNode *target() const { return m_target; }
	ASTNode *value() const { return m_value; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class Comprehension : public ASTNode
{
	ASTNode *m_target{ nullptr };
	ASTNode *m_iter{ nullptr };
	std::vector<ASTNode *> m_ifs;
	const bool m_is_async;

  public:
	Comprehension(ASTNode *target,
		ASTNode *iter,
		std::vector<ASTNode *> ifs,
		bool is_async,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::Comprehension, source_location), m_target(target), m_iter(iter),
		  m_ifs(ifs), m_is_async(is_async)
	{}

	ASTNode *target() const { return m_target; }
	ASTNode *iter() const { return m_iter; }
	const std::vector<ASTNode *> &ifs() const { return m_ifs; }
	std::vector<ASTNode *> &ifs() { return m_ifs; }

	bool is_async() const { return m_is_async; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class ListComp : public ASTNode
{
	ASTNode *m_elt{ nullptr };
	std::vector<Comprehension *> m_generators;

  public:
	ListComp(ASTNode *elt,
		std::vector<Comprehension *> &&generators,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::ListComp, source_location), m_elt(std::move(elt)),
		  m_generators(std::move(generators))
	{}

	ASTNode *elt() const { return m_elt; }
	const std::vector<Comprehension *> &generators() const { return m_generators; }
	std::vector<Comprehension *> &generators() { return m_generators; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class DictComp : public ASTNode
{
	ASTNode *m_key{ nullptr };
	ASTNode *m_value{ nullptr };
	std::vector<Comprehension *> m_generators;

  public:
	DictComp(ASTNode *key,
		ASTNode *value,
		std::vector<Comprehension *> &&generators,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::DictComp, source_location), m_key(std::move(key)),
		  m_value(std::move(value)), m_generators(std::move(generators))
	{}

	ASTNode *key() const { return m_key; }
	ASTNode *value() const { return m_value; }
	const std::vector<Comprehension *> &generators() const { return m_generators; }
	std::vector<Comprehension *> &generators() { return m_generators; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class GeneratorExp : public ASTNode
{
	ASTNode *m_elt{ nullptr };
	std::vector<Comprehension *> m_generators;

  public:
	GeneratorExp(ASTNode *elt,
		std::vector<Comprehension *> &&generators,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::GeneratorExp, source_location), m_elt(std::move(elt)),
		  m_generators(std::move(generators))
	{}

	ASTNode *elt() const { return m_elt; }
	const std::vector<Comprehension *> &generators() const { return m_generators; }
	std::vector<Comprehension *> &generators() { return m_generators; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class SetComp : public ASTNode
{
	ASTNode *m_elt{ nullptr };
	std::vector<Comprehension *> m_generators;

  public:
	SetComp(ASTNode *elt, std::vector<Comprehension *> &&generators, SourceLocation source_location)
		: ASTNode(ASTNodeType::SetComp, source_location), m_elt(std::move(elt)),
		  m_generators(std::move(generators))
	{}

	ASTNode *elt() const { return m_elt; }
	const std::vector<Comprehension *> &generators() const { return m_generators; }
	std::vector<Comprehension *> &generators() { return m_generators; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class JoinedStr : public ASTNode
{
	std::vector<ASTNode *> m_values;

  public:
	JoinedStr(std::vector<ASTNode *> values, SourceLocation source_location)
		: ASTNode(ASTNodeType::JoinedStr, source_location), m_values(std::move(values))
	{}

	const std::vector<ASTNode *> &values() const { return m_values; }
	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

class FormattedValue : public ASTNode
{
  public:
	enum class Conversion { NONE = 0, REPR = 1, STRING = 2, ASCII = 3 };

  private:
	ASTNode *m_value{ nullptr };
	Conversion m_conversion;
	JoinedStr *m_format_spec{ nullptr };

  public:
	FormattedValue(ASTNode *value,
		Conversion conversion,
		JoinedStr *format_spec,
		SourceLocation source_location)
		: ASTNode(ASTNodeType::FormattedValue, source_location), m_value(std::move(value)),
		  m_conversion(conversion), m_format_spec(std::move(format_spec))
	{}

	ASTNode *value() const { return m_value; }
	Conversion conversion() const { return m_conversion; }
	JoinedStr *format_spec() const { return m_format_spec; }

	Value *codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


template<typename NodeType> NodeType *as(ASTNode *node);
template<typename NodeType> const NodeType *as(const ASTNode *node);

#define __AST_NODE_TYPE(x)           \
	template<> x *as(ASTNode *node); \
	template<> const x *as(const ASTNode *node);
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

// TODO: re-port to arena ownership and re-enable. Disabled during the
// shared_ptr -> arena migration of AST nodes; only ConstantFolding and
// its tests depend on this visitor, and they are excluded from the build.
#if 0
struct NodeTransformVisitor
{
	virtual ~NodeTransformVisitor() = default;

	bool m_can_return_multiple_nodes{ false };
#define __AST_NODE_TYPE(NodeType) virtual std::vector<ASTNode *> visit(NodeType *node);
	AST_NODE_TYPES
#undef __AST_NODE_TYPE

  protected:
	void transform_single_node(ASTNode * node);

	void transform_multiple_nodes(std::vector<ASTNode *> &nodes);
};
#endif


}// namespace ast
