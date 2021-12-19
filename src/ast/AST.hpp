#pragma once

#include <memory>
#include <numeric>
#include <optional>
#include <stack>
#include <string>
#include <vector>

#include "forward.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"
#include "utilities.hpp"

#include "spdlog/spdlog.h"

namespace ast {

#define AST_NODE_TYPES                  \
	__AST_NODE_TYPE(Argument)           \
	__AST_NODE_TYPE(Arguments)          \
	__AST_NODE_TYPE(Attribute)          \
	__AST_NODE_TYPE(Assign)             \
	__AST_NODE_TYPE(Assert)             \
	__AST_NODE_TYPE(AugAssign)          \
	__AST_NODE_TYPE(BinaryExpr)         \
	__AST_NODE_TYPE(BoolOp)             \
	__AST_NODE_TYPE(Call)               \
	__AST_NODE_TYPE(ClassDefinition)    \
	__AST_NODE_TYPE(Compare)            \
	__AST_NODE_TYPE(Constant)           \
	__AST_NODE_TYPE(Dict)               \
	__AST_NODE_TYPE(ExceptHandler)      \
	__AST_NODE_TYPE(For)                \
	__AST_NODE_TYPE(FunctionDefinition) \
	__AST_NODE_TYPE(If)                 \
	__AST_NODE_TYPE(Import)             \
	__AST_NODE_TYPE(Keyword)            \
	__AST_NODE_TYPE(List)               \
	__AST_NODE_TYPE(Module)             \
	__AST_NODE_TYPE(Name)               \
	__AST_NODE_TYPE(Raise)              \
	__AST_NODE_TYPE(Return)             \
	__AST_NODE_TYPE(Subscript)          \
	__AST_NODE_TYPE(Try)                \
	__AST_NODE_TYPE(Tuple)              \
	__AST_NODE_TYPE(UnaryExpr)          \
	__AST_NODE_TYPE(While)


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
	ASSERT_NOT_REACHED()
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

  private:
	virtual void print_this_node(const std::string &indent) const = 0;
	virtual void print_this_node() const { print_this_node(""); };

  public:
	ASTNode(ASTNodeType node_type) : m_node_type(node_type) {}
	void print_node(const std::string &indent) { print_this_node(indent); }
	ASTNodeType node_type() const { return m_node_type; }
	virtual ~ASTNode() = default;

	virtual void codegen(CodeGenerator *) const = 0;
};// namespace ast

// FormatterValue, JoinedStr, List, Tuple, Set, Dict

class Constant : public ASTNode
{
	Value m_value;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	explicit Constant(Value value) : ASTNode(ASTNodeType::Constant), m_value(value) {}
	explicit Constant(double value) : ASTNode(ASTNodeType::Constant), m_value(Number{ value }) {}
	explicit Constant(int64_t value) : ASTNode(ASTNodeType::Constant), m_value(Number{ value }) {}
	explicit Constant(bool value) : ASTNode(ASTNodeType::Constant), m_value(NameConstant{ value })
	{}
	explicit Constant(NoneType value)
		: ASTNode(ASTNodeType::Constant), m_value(NameConstant{ value })
	{}
	explicit Constant(std::string value)
		: ASTNode(ASTNodeType::Constant), m_value(String{ std::move(value) })
	{}
	explicit Constant(const char *value)
		: ASTNode(ASTNodeType::Constant), m_value(String{ std::string(value) })
	{}

	const Value &value() const { return m_value; }
	Value value() { return m_value; }

	virtual void codegen(CodeGenerator *) const override;
};


class List : public ASTNode
{
  private:
	std::vector<std::shared_ptr<ASTNode>> m_elements;
	ContextType m_ctx;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	List(std::vector<std::shared_ptr<ASTNode>> elements, ContextType ctx)
		: ASTNode(ASTNodeType::List), m_elements(std::move(elements)), m_ctx(ctx)
	{}

	List(ContextType ctx) : ASTNode(ASTNodeType::List), m_elements(), m_ctx(ctx) {}

	void append(std::shared_ptr<ASTNode> element) { m_elements.push_back(std::move(element)); }

	ContextType context() const { return m_ctx; }
	const std::vector<std::shared_ptr<ASTNode>> &elements() const { return m_elements; }

	void codegen(CodeGenerator *) const override;
};

class Tuple : public ASTNode
{
  private:
	std::vector<std::shared_ptr<ASTNode>> m_elements;
	ContextType m_ctx;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Tuple(std::vector<std::shared_ptr<ASTNode>> elements, ContextType ctx)
		: ASTNode(ASTNodeType::Tuple), m_elements(std::move(elements)), m_ctx(ctx)
	{}

	Tuple(ContextType ctx) : ASTNode(ASTNodeType::Tuple), m_elements(), m_ctx(ctx) {}

	void append(std::shared_ptr<ASTNode> element) { m_elements.push_back(std::move(element)); }

	ContextType context() const { return m_ctx; }
	const std::vector<std::shared_ptr<ASTNode>> &elements() const { return m_elements; }
	std::vector<std::shared_ptr<ASTNode>> &elements() { return m_elements; }

	void codegen(CodeGenerator *) const override;
};


class Dict : public ASTNode
{
  private:
	std::vector<std::shared_ptr<ASTNode>> m_keys;
	std::vector<std::shared_ptr<ASTNode>> m_values;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Dict(std::vector<std::shared_ptr<ASTNode>> keys, std::vector<std::shared_ptr<ASTNode>> values)
		: ASTNode(ASTNodeType::Dict), m_keys(std::move(keys)), m_values(std::move(values))
	{}

	Dict() : ASTNode(ASTNodeType::Dict), m_keys(), m_values() {}

	void insert(std::shared_ptr<ASTNode> key, std::shared_ptr<ASTNode> value)
	{
		m_keys.push_back(std::move(key));
		m_values.push_back(std::move(value));
	}

	const std::vector<std::shared_ptr<ASTNode>> &keys() const { return m_keys; }
	const std::vector<std::shared_ptr<ASTNode>> &values() const { return m_values; }

	void codegen(CodeGenerator *) const override;
};


class Variable : public ASTNode
{
  public:
	ContextType context_type() const { return m_ctx; }

	virtual const std::vector<std::string> &ids() const = 0;

  protected:
	ContextType m_ctx;

  protected:
	Variable(ASTNodeType node_type, ContextType ctx) : ASTNode(node_type), m_ctx(ctx) {}
};


class Name final : public Variable
{
	std::vector<std::string> m_id;

  private:
	void print_this_node(const std::string &indent) const override;

  public:
	Name(std::string id, ContextType ctx)
		: Variable(ASTNodeType::Name, ctx), m_id({ std::move(id) })
	{}

	const std::vector<std::string> &ids() const final { return m_id; }
	void set_context(ContextType ctx) { m_ctx = ctx; }

	void codegen(CodeGenerator *) const override;
};


class Statement : public ASTNode
{
  public:
	Statement(ASTNodeType node_type) : ASTNode(node_type) {}
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
		std::string type_comment)
		: Statement(ASTNodeType::Assign), m_targets(std::move(targets)), m_value(std::move(value)),
		  m_type_comment(std::move(type_comment))
	{}

	const std::vector<std::shared_ptr<ASTNode>> &targets() const { return m_targets; }
	const std::shared_ptr<ASTNode> &value() const { return m_value; }
	void set_value(std::shared_ptr<ASTNode> v) { m_value = std::move(v); }

	void codegen(CodeGenerator *) const override;
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
	ASSERT_NOT_REACHED()
}

class UnaryExpr : public ASTNode
{
  public:
  private:
	const UnaryOpType m_op_type;
	std::shared_ptr<ASTNode> m_operand;

  public:
	UnaryExpr(UnaryOpType op_type, std::shared_ptr<ASTNode> operand)
		: ASTNode(ASTNodeType::UnaryExpr), m_op_type(op_type), m_operand(std::move(operand))
	{}

	const std::shared_ptr<ASTNode> &operand() const { return m_operand; }
	std::shared_ptr<ASTNode> &operand() { return m_operand; }

	UnaryOpType op_type() const { return m_op_type; }

	void codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};

#define BINARY_OPERATIONS  \
	__BINARY_OP(PLUS)      \
	__BINARY_OP(MINUS)     \
	__BINARY_OP(MODULO)    \
	__BINARY_OP(MULTIPLY)  \
	__BINARY_OP(EXP)       \
	__BINARY_OP(SLASH)     \
	__BINARY_OP(LEFTSHIFT) \
	__BINARY_OP(RIGHTSHIFT)

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
	ASSERT_NOT_REACHED()
}

class BinaryExpr : public ASTNode
{
  public:
  private:
	const BinaryOpType m_op_type;
	std::shared_ptr<ASTNode> m_lhs;
	std::shared_ptr<ASTNode> m_rhs;

  public:
	BinaryExpr(BinaryOpType op_type, std::shared_ptr<ASTNode> lhs, std::shared_ptr<ASTNode> rhs)
		: ASTNode(ASTNodeType::BinaryExpr), m_op_type(op_type), m_lhs(std::move(lhs)),
		  m_rhs(std::move(rhs))
	{}

	const std::shared_ptr<ASTNode> &lhs() const { return m_lhs; }
	std::shared_ptr<ASTNode> &lhs() { return m_lhs; }

	const std::shared_ptr<ASTNode> &rhs() const { return m_rhs; }
	std::shared_ptr<ASTNode> &rhs() { return m_rhs; }

	BinaryOpType op_type() const { return m_op_type; }

	void codegen(CodeGenerator *) const override;

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
	AugAssign(std::shared_ptr<ASTNode> target, BinaryOpType op, std::shared_ptr<ASTNode> value)
		: Statement(ASTNodeType::AugAssign), m_target(std::move(target)), m_op(op),
		  m_value(std::move(value))
	{}

	const std::shared_ptr<ASTNode> &target() const { return m_target; }
	BinaryOpType op() const { return m_op; }
	const std::shared_ptr<ASTNode> &value() const { return m_value; }
	void set_value(std::shared_ptr<ASTNode> value) { m_value = std::move(value); }

	void codegen(CodeGenerator *) const override;
};

class Return : public ASTNode
{
	std::shared_ptr<ASTNode> m_value;

  public:
	Return(std::shared_ptr<ASTNode> value) : ASTNode(ASTNodeType::Return), m_value(std::move(value))
	{}

	std::shared_ptr<ASTNode> value() const { return m_value; }

	void print_this_node(const std::string &indent) const override;

	void codegen(CodeGenerator *) const override;
};


class Argument final : public ASTNode
{
	const std::string m_arg;
	const std::string m_annotation;
	const std::string m_type_comment;

  public:
	Argument(std::string arg, std::string annotation, std::string type_comment)
		: ASTNode(ASTNodeType::Argument), m_arg(std::move(arg)),
		  m_annotation(std::move(annotation)), m_type_comment(std::move(type_comment))
	{}

	void print_this_node(const std::string &indent) const final;

	const std::string &name() const { return m_arg; }

	void codegen(CodeGenerator *) const override;
};


class Arguments : public ASTNode
{
	std::vector<std::shared_ptr<Argument>> m_posonlyargs;
	std::vector<std::shared_ptr<Argument>> m_args;
	std::vector<std::shared_ptr<Argument>> m_kwargs;
	std::shared_ptr<Argument> m_vararg;
	std::shared_ptr<Argument> m_kwarg;
	std::vector<std::shared_ptr<Constant>> m_kw_defaults;
	std::vector<std::shared_ptr<Constant>> m_defaults;

  public:
	Arguments() : ASTNode(ASTNodeType::Arguments) {}
	Arguments(std::vector<std::shared_ptr<Argument>> args) : Arguments()
	{
		m_args = std::move(args);
	}


	void print_this_node(const std::string &indent) const final;

	void push_arg(std::shared_ptr<Argument> arg) { m_args.push_back(std::move(arg)); }
	std::vector<std::string> argument_names() const;

	void push_kwarg(std::shared_ptr<Argument> arg) { m_kwargs.push_back(std::move(arg)); }
	std::vector<std::string> keyword_argument_names() const;

	const std::vector<std::shared_ptr<Argument>> &args() const { return m_args; }

	void codegen(CodeGenerator *) const override;
};

class FunctionDefinition final : public ASTNode
{
	const std::string m_function_name;
	const std::shared_ptr<Arguments> m_args;
	const std::vector<std::shared_ptr<ASTNode>> m_body;
	const std::vector<std::shared_ptr<ASTNode>> m_decorator_list;
	const std::shared_ptr<ASTNode> m_returns;
	std::string m_type_comment;

	void print_this_node(const std::string &indent) const final;

  public:
	FunctionDefinition(std::string function_name,
		std::shared_ptr<Arguments> args,
		std::vector<std::shared_ptr<ASTNode>> body,
		std::vector<std::shared_ptr<ASTNode>> decorator_list,
		std::shared_ptr<ASTNode> returns,
		std::string type_comment)
		: ASTNode(ASTNodeType::FunctionDefinition), m_function_name(std::move(function_name)),
		  m_args(std::move(args)), m_body(std::move(body)),
		  m_decorator_list(std::move(decorator_list)), m_returns(std::move(returns)),
		  m_type_comment(std::move(type_comment))
	{}

	const std::string &name() const { return m_function_name; }
	const std::shared_ptr<Arguments> &args() const { return m_args; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &decorator_list() const { return m_decorator_list; }
	const std::shared_ptr<ASTNode> &returns() const { return m_returns; }
	const std::string &type_comment() const { return m_type_comment; }

	void codegen(CodeGenerator *) const override;
};


class Keyword : public ASTNode
{
	const std::string m_arg;
	std::shared_ptr<ASTNode> m_value;

  public:
	Keyword(std::string arg, std::shared_ptr<ASTNode> value)
		: ASTNode(ASTNodeType::Keyword), m_arg(std::move(arg))
	{
		m_value = std::move(value);
	}

	void print_this_node(const std::string &indent) const final;

	const std::string &arg() const { return m_arg; }
	std::shared_ptr<ASTNode> value() const { return m_value; }

	void codegen(CodeGenerator *) const override;
};


class ClassDefinition final : public ASTNode
{
	const std::string m_class_name;
	const std::vector<std::shared_ptr<ASTNode>> m_bases;
	const std::vector<std::shared_ptr<Keyword>> m_keywords;
	const std::vector<std::shared_ptr<ASTNode>> m_body;
	const std::vector<std::shared_ptr<ASTNode>> m_decorator_list;

	void print_this_node(const std::string &indent) const final;

  public:
	ClassDefinition(std::string class_name,
		std::vector<std::shared_ptr<ASTNode>> bases,
		std::vector<std::shared_ptr<Keyword>> keywords,
		std::vector<std::shared_ptr<ASTNode>> body,
		std::vector<std::shared_ptr<ASTNode>> decorator_list)
		: ASTNode(ASTNodeType::ClassDefinition), m_class_name(std::move(class_name)),
		  m_bases(std::move(bases)), m_keywords(std::move(keywords)), m_body(std::move(body)),
		  m_decorator_list(std::move(decorator_list))
	{}

	const std::string &name() const { return m_class_name; }
	const std::vector<std::shared_ptr<ASTNode>> &bases() const { return m_bases; }
	const std::vector<std::shared_ptr<Keyword>> &keywords() const { return m_keywords; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &decorator_list() const { return m_decorator_list; }

	void codegen(CodeGenerator *) const override;
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
		std::vector<std::shared_ptr<Keyword>> keywords)
		: ASTNode(ASTNodeType::Call), m_function(std::move(function)), m_args(std::move(args)),
		  m_keywords(std::move(keywords))
	{}

	Call(std::shared_ptr<ASTNode> function) : Call(function, {}, {}) {}

	const std::shared_ptr<ASTNode> &function() const { return m_function; }
	const std::vector<std::shared_ptr<ASTNode>> &args() const { return m_args; }
	const std::vector<std::shared_ptr<Keyword>> &keywords() const { return m_keywords; }

	void codegen(CodeGenerator *) const override;
};

class Module : public ASTNode
{
	std::string m_filename;
	std::vector<std::shared_ptr<ASTNode>> m_body;

  public:
	Module(std::string filename) : ASTNode(ASTNodeType::Module), m_filename(std::move(filename)) {}

	template<typename T> void emplace(T node) { m_body.emplace_back(std::move(node)); }

	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	std::vector<std::shared_ptr<ASTNode>> &body() { return m_body; }

	const std::string &filename() const { return m_filename; }

	void codegen(CodeGenerator *) const override;

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
		std::vector<std::shared_ptr<ASTNode>> orelse)
		: ASTNode(ASTNodeType::If), m_test(std::move(test)), m_body(std::move(body)),
		  m_orelse(std::move(orelse))
	{}

	const std::shared_ptr<ASTNode> &test() const { return m_test; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &orelse() const { return m_orelse; }

	void codegen(CodeGenerator *) const override;

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
		std::string type_comment)
		: ASTNode(ASTNodeType::For), m_target(std::move(target)), m_iter(std::move(iter)),
		  m_body(std::move(body)), m_orelse(std::move(orelse)), m_type_comment(type_comment)
	{}

	const std::shared_ptr<ASTNode> &target() const { return m_target; }
	const std::shared_ptr<ASTNode> &iter() const { return m_iter; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &orelse() const { return m_orelse; }
	const std::string &type_comment() const { return m_type_comment; }

	void codegen(CodeGenerator *) const override;

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
		std::vector<std::shared_ptr<ASTNode>> orelse)
		: ASTNode(ASTNodeType::While), m_test(std::move(test)), m_body(std::move(body)),
		  m_orelse(std::move(orelse))
	{}

	const std::shared_ptr<ASTNode> &test() const { return m_test; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &orelse() const { return m_orelse; }

	void codegen(CodeGenerator *) const override;

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
	OpType m_op;
	std::shared_ptr<ASTNode> m_rhs;

  public:
	Compare(std::shared_ptr<ASTNode> lhs, OpType op, std::shared_ptr<ASTNode> rhs)
		: ASTNode(ASTNodeType::Compare), m_lhs(std::move(lhs)), m_op(op), m_rhs(std::move(rhs))
	{}

	const std::shared_ptr<ASTNode> &lhs() const { return m_lhs; }
	OpType op() const { return m_op; }
	const std::shared_ptr<ASTNode> &rhs() const { return m_rhs; }

	void codegen(CodeGenerator *) const override;

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
		ASSERT_NOT_REACHED()
	}

	void print_this_node(const std::string &indent) const override;
};


class Attribute : public ASTNode
{
	std::shared_ptr<ASTNode> m_value;
	std::string m_attr;
	ContextType m_ctx;

  public:
	Attribute(std::shared_ptr<ASTNode> value, std::string attr, ContextType ctx)
		: ASTNode(ASTNodeType::Attribute), m_value(std::move(value)), m_attr(std::move(attr)),
		  m_ctx(ctx)
	{}

	const std::shared_ptr<ASTNode> &value() const { return m_value; }
	const std::string &attr() const { return m_attr; }
	ContextType context() const { return m_ctx; }

	void codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class Import : public ASTNode
{
	std::vector<std::string> m_names;
	std::optional<std::string> m_asname;

  public:
	Import() : ASTNode(ASTNodeType::Import) {}

	Import(std::vector<std::string> names) : ASTNode(ASTNodeType::Import), m_names(std::move(names))
	{}

	Import(std::vector<std::string> names, std::optional<std::string> asname)
		: ASTNode(ASTNodeType::Import), m_names(std::move(names)), m_asname(std::move(asname))
	{}

	const std::optional<std::string> &asname() const { return m_asname; }
	const std::vector<std::string> &names() const { return m_names; }
	std::string dotted_name() const
	{
		return std::accumulate(
			std::next(m_names.begin()), m_names.end(), *m_names.begin(), [](auto rhs, auto lhs) {
				return std::move(rhs) + "." + lhs;
			});
	}
	void set_asname(std::string name) { m_asname = std::move(name); }

	void add_dotted_name(std::string name) { m_names.push_back(std::move(name)); }

	void codegen(CodeGenerator *) const override;

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
	SliceType m_slice;
	ContextType m_ctx;

  public:
	Subscript() : ASTNode(ASTNodeType::Subscript) {}

	Subscript(std::shared_ptr<ASTNode> value, SliceType slice, ContextType ctx)
		: ASTNode(ASTNodeType::Subscript), m_value(std::move(value)), m_slice(std::move(slice)),
		  m_ctx(ctx)
	{}

	const std::shared_ptr<ASTNode> &value() const { return m_value; }
	const SliceType &slice() const { return m_slice; }
	ContextType context() const { return m_ctx; }

	void set_value(std::shared_ptr<ASTNode> value) { m_value = std::move(value); }
	void set_slice(SliceType slice) { m_slice = std::move(slice); }
	void set_context(ContextType context) { m_ctx = context; }

	void codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class Raise : public ASTNode
{
  public:
	std::shared_ptr<ASTNode> m_exception;
	std::shared_ptr<ASTNode> m_cause;

  public:
	Raise() : ASTNode(ASTNodeType::Raise), m_cause(std::make_shared<Constant>(NoneType{})) {}

	Raise(std::shared_ptr<ASTNode> exception, std::shared_ptr<ASTNode> cause)
		: ASTNode(ASTNodeType::Raise), m_exception(std::move(exception)), m_cause(std::move(cause))
	{}

	const std::shared_ptr<ASTNode> &exception() const { return m_exception; }
	const std::shared_ptr<ASTNode> &cause() const { return m_cause; }

	void codegen(CodeGenerator *) const override;

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
		std::vector<std::shared_ptr<ASTNode>> body)
		: ASTNode(ASTNodeType::ExceptHandler), m_type(std::move(type)), m_name(std::move(name)),
		  m_body(std::move(body))
	{}

	const std::shared_ptr<ASTNode> &type() const { return m_type; }
	const std::string &name() const { return m_name; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }

	void codegen(CodeGenerator *) const override;

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
		std::vector<std::shared_ptr<ASTNode>> finalbody)
		: ASTNode(ASTNodeType::Try), m_body(std::move(body)), m_handlers(std::move(handlers)),
		  m_orelse(std::move(orelse)), m_finalbody(std::move(finalbody))
	{}

	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	const std::vector<std::shared_ptr<ExceptHandler>> &handlers() const { return m_handlers; }
	const std::vector<std::shared_ptr<ASTNode>> &orelse() const { return m_orelse; }
	const std::vector<std::shared_ptr<ASTNode>> &cause() const { return m_finalbody; }

	void codegen(CodeGenerator *) const override;

  private:
	void print_this_node(const std::string &indent) const override;
};


class Assert : public ASTNode
{
  public:
	std::shared_ptr<ASTNode> m_test{ nullptr };
	std::shared_ptr<ASTNode> m_msg{ nullptr };

  public:
	Assert(std::shared_ptr<ASTNode> test, std::shared_ptr<ASTNode> msg)
		: ASTNode(ASTNodeType::Assert), m_test(std::move(test)), m_msg(std::move(msg))
	{
		ASSERT(m_test)
	}

	const std::shared_ptr<ASTNode> &test() const { return m_test; }
	const std::shared_ptr<ASTNode> &msg() const { return m_msg; }

	void codegen(CodeGenerator *) const override;

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
	BoolOp(OpType op, std::vector<std::shared_ptr<ASTNode>> values)
		: ASTNode(ASTNodeType::BoolOp), m_op(op), m_values(std::move(values))
	{
		ASSERT(m_values.size() >= 2);
	}

	OpType op() const { return m_op; }
	const std::vector<std::shared_ptr<ASTNode>> &values() const { return m_values; }

	void codegen(CodeGenerator *) const override;

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
		ASSERT_NOT_REACHED()
	}

	void print_this_node(const std::string &indent) const override;
};


template<typename NodeType> std::shared_ptr<NodeType> as(std::shared_ptr<ASTNode> node);

#define __AST_NODE_TYPE(x) template<> std::shared_ptr<x> as(std::shared_ptr<ASTNode> node);
AST_NODE_TYPES
#undef __AST_NODE_TYPE

struct CodeGenerator
{
#define __AST_NODE_TYPE(NodeType) virtual void visit(const NodeType *node) = 0;
	AST_NODE_TYPES
#undef __AST_NODE_TYPE
};

}// namespace ast