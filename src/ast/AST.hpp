#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstddef>
#include <optional>
#include <stack>

#include "forward.hpp"
#include "utilities.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"

#include "spdlog/spdlog.h"

namespace ast {

#define AST_NODE_TYPES                  \
	__AST_NODE_TYPE(Argument)           \
	__AST_NODE_TYPE(Arguments)          \
	__AST_NODE_TYPE(Assign)             \
	__AST_NODE_TYPE(BinaryExpr)         \
	__AST_NODE_TYPE(Call)               \
	__AST_NODE_TYPE(Compare)            \
	__AST_NODE_TYPE(Constant)           \
	__AST_NODE_TYPE(For)                \
	__AST_NODE_TYPE(FunctionDefinition) \
	__AST_NODE_TYPE(If)                 \
	__AST_NODE_TYPE(List)               \
	__AST_NODE_TYPE(Module)             \
	__AST_NODE_TYPE(Name)               \
	__AST_NODE_TYPE(Return)

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
}

#define __AST_NODE_TYPE(x) class x;
AST_NODE_TYPES
#undef __AST_NODE_TYPE


class ASTContext
{
	std::stack<std::shared_ptr<Arguments>> m_local_args;

  public:
	void push_local_args(std::shared_ptr<Arguments> args) { m_local_args.push(std::move(args)); }

	void pop_local_args() { m_local_args.pop(); }

	bool has_local_args() const { return !m_local_args.empty(); }

	const std::shared_ptr<Arguments> &local_args() const { return m_local_args.top(); }
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
	virtual Register generate(size_t function_id, BytecodeGenerator &, ASTContext &) const = 0;

	template<typename FuncType> void visit(FuncType &&func) const;
};// namespace ast

// FormatterValue, JoinedStr, List, Tuple, Set, Dict

class Constant : public ASTNode
{
	Value m_value;

  private:
	void print_this_node(const std::string &indent) const override
	{
		spdlog::debug("{}Constant", indent);
		std::visit(overloaded{ [&indent](const auto &value) {
								  spdlog::debug("{}  - value: {}", indent, value.to_string());
							  },
					   [&indent](const std::shared_ptr<PyObject> &value) {
						   spdlog::debug("{}  - value: {}", indent, value->to_string());
					   } },
			m_value);
	}

  public:
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

	Register generate(size_t function_id, BytecodeGenerator &generator, ASTContext &) const final;
};


class List : public ASTNode
{
  public:
	enum class ContextType { LOAD = 0, STORE = 1, UNSET = 2 };

  private:
	std::vector<std::shared_ptr<ASTNode>> m_elements;
	ContextType m_ctx;

  private:
	void print_this_node(const std::string &indent) const override
	{
		spdlog::debug("{}List", indent);
		spdlog::debug("{}  context: {}", indent, static_cast<int>(m_ctx));
		spdlog::debug("{}  elements:", indent);
		std::string new_indent = indent + std::string(6, ' ');
		for (const auto &el : m_elements) { el->print_node(new_indent); }
	}

  public:
	List(std::vector<std::shared_ptr<ASTNode>> elements, ContextType ctx)
		: ASTNode(ASTNodeType::List), m_elements(std::move(elements)), m_ctx(ctx)
	{}

	List(ContextType ctx) : ASTNode(ASTNodeType::List), m_elements(), m_ctx(ctx) {}

	void append(std::shared_ptr<ASTNode> element) { m_elements.push_back(std::move(element)); }

	ContextType context() const { return m_ctx; }
	const std::vector<std::shared_ptr<ASTNode>> &elements() const { return m_elements; }

	Register generate(size_t, BytecodeGenerator &, ASTContext &) const final;
};


class Variable : public ASTNode
{
  public:
	enum class ContextType { LOAD = 0, STORE = 1, DEL = 2, UNSET = 3 };
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
	void print_this_node(const std::string &indent) const override
	{
		spdlog::debug("{}Name", indent);
		spdlog::debug("{}  - id: \"{}\"", indent, m_id[0]);
		spdlog::debug("{}  - context_type: {}", indent, static_cast<int>(m_ctx));
	}

  public:
	Name(std::string id, ContextType ctx)
		: Variable(ASTNodeType::Name, ctx), m_id({ std::move(id) })
	{}

	const std::vector<std::string> &ids() const final { return m_id; }

	Register generate(size_t function_id, BytecodeGenerator &, ASTContext &) const final;
};


class Statement : public ASTNode
{
  public:
	Statement(ASTNodeType node_type) : ASTNode(node_type) {}
};

class Assign : public Statement
{
	std::vector<std::shared_ptr<Variable>> m_targets;
	std::shared_ptr<ASTNode> m_value;
	std::string m_type_comment;

  private:
	void print_this_node(const std::string &indent) const override
	{
		spdlog::debug("{}Assign", indent);
		spdlog::debug("{}  - targets:", indent);
		std::string new_indent = indent + std::string(6, ' ');
		for (const auto &t : m_targets) { t->print_node(new_indent); }
		spdlog::debug("{}  - value:", indent);
		m_value->print_node(new_indent);
		spdlog::debug("{}  - comment type: {}", indent, m_type_comment);
	}

  public:
	Assign(std::vector<std::shared_ptr<Variable>> targets,
		std::shared_ptr<ASTNode> value,
		std::string type_comment)
		: Statement(ASTNodeType::Assign), m_targets(std::move(targets)), m_value(std::move(value)),
		  m_type_comment(std::move(type_comment))
	{}

	const std::vector<std::shared_ptr<Variable>> &targets() const { return m_targets; }
	const std::shared_ptr<ASTNode> &value() const { return m_value; }

	Register generate(size_t function_id, BytecodeGenerator &generator, ASTContext &) const final;
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

class BinaryExpr : public ASTNode
{
  public:
	enum class OpType {
#define __BINARY_OP(x) x,
		BINARY_OPERATIONS
#undef __BINARY_OP
	};

  private:
	const OpType m_op_type;
	std::shared_ptr<ASTNode> m_lhs;
	std::shared_ptr<ASTNode> m_rhs;

  public:
	BinaryExpr(OpType op_type, std::shared_ptr<ASTNode> lhs, std::shared_ptr<ASTNode> rhs)
		: ASTNode(ASTNodeType::BinaryExpr), m_op_type(op_type), m_lhs(std::move(lhs)),
		  m_rhs(std::move(rhs))
	{}

	const std::shared_ptr<ASTNode> &lhs() const { return m_lhs; }
	std::shared_ptr<ASTNode> &lhs() { return m_lhs; }

	const std::shared_ptr<ASTNode> &rhs() const { return m_rhs; }
	std::shared_ptr<ASTNode> &rhs() { return m_rhs; }

	OpType op_type() const { return m_op_type; }

	Register generate(size_t function_id, BytecodeGenerator &, ASTContext &) const final;

  private:
	std::string_view op_type_to_string(OpType type) const
	{
		switch (type) {
#define __BINARY_OP(x) \
	case OpType::x:    \
		return #x;
			BINARY_OPERATIONS
#undef __BINARY_OP
		}
		ASSERT_NOT_REACHED()
	}

	void print_this_node(const std::string &indent) const override
	{
		spdlog::debug("{}BinaryOp", indent);
		spdlog::debug("{}  - op_type: {}", indent, op_type_to_string(m_op_type));
		spdlog::debug("{}  - lhs:", indent);
		std::string new_indent = indent + std::string(6, ' ');
		m_lhs->print_node(new_indent);
		spdlog::debug("{}  - rhs:", indent);
		m_rhs->print_node(new_indent);
	}
};

class Return : public ASTNode
{
	std::shared_ptr<ASTNode> m_value;

  public:
	Return(std::shared_ptr<ASTNode> value) : ASTNode(ASTNodeType::Return), m_value(std::move(value))
	{}

	std::shared_ptr<ASTNode> value() const { return m_value; }

	Register generate(size_t, BytecodeGenerator &, ASTContext &) const final;

	void print_this_node(const std::string &indent) const override
	{
		spdlog::debug("{}Return", indent);
		spdlog::debug("{}  - value:", indent);
		std::string new_indent = indent + std::string(6, ' ');
		m_value->print_node(new_indent);
	}
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

	void print_this_node(const std::string &indent) const final
	{
		spdlog::debug("{}Argument", indent);
		spdlog::debug("{}  - arg: {}", indent, m_arg);
		spdlog::debug("{}  - annotation: {}", indent, m_annotation);
		spdlog::debug("{}  - type_comment: {}", indent, m_type_comment);
	}
	const std::string &name() const { return m_arg; }

	Register generate(size_t, BytecodeGenerator &, ASTContext &) const final;
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


	void print_this_node(const std::string &indent) const final
	{
		spdlog::debug("{}Arguments", indent);
		spdlog::debug("{}  - args:", indent);
		std::string new_indent = indent + std::string(6, ' ');
		for (const auto &arg : m_args) { arg->print_node(new_indent); }
	}

	void push_arg(std::shared_ptr<Argument> arg) { m_args.push_back(std::move(arg)); }
	std::vector<std::string> argument_names() const;

	Register generate(size_t, BytecodeGenerator &, ASTContext &) const final;
};

class FunctionDefinition final : public ASTNode
{
	const std::string m_function_name;
	const std::shared_ptr<Arguments> m_args;
	const std::vector<std::shared_ptr<ASTNode>> m_body;
	const std::vector<std::shared_ptr<ASTNode>> m_decorator_list;
	const std::shared_ptr<ASTNode> m_returns;
	std::string m_type_comment;

	void print_this_node(const std::string &indent) const final
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

	Register generate(size_t function_id, BytecodeGenerator &, ASTContext &) const final;
};


class Call : public ASTNode
{
	std::shared_ptr<ASTNode> m_function;
	std::vector<std::shared_ptr<ASTNode>> m_args;
	std::vector<std::shared_ptr<ASTNode>> m_keywords;

	void print_this_node(const std::string &indent) const final
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

  public:
	Call(std::shared_ptr<ASTNode> function,
		std::vector<std::shared_ptr<ASTNode>> args,
		std::vector<std::shared_ptr<ASTNode>> keywords)
		: ASTNode(ASTNodeType::Call), m_function(std::move(function)), m_args(std::move(args)),
		  m_keywords(std::move(keywords))
	{}

	Call(std::shared_ptr<ASTNode> function) : Call(function, {}, {}) {}

	const std::shared_ptr<ASTNode> &function() const { return m_function; }
	const std::vector<std::shared_ptr<ASTNode>> &args() const { return m_args; }
	const std::vector<std::shared_ptr<ASTNode>> &keywords() const { return m_keywords; }

	Register generate(size_t, BytecodeGenerator &, ASTContext &) const final;
};

class Module : public ASTNode
{
	std::vector<std::shared_ptr<ASTNode>> m_body;

  public:
	Module() : ASTNode(ASTNodeType::Module) {}

	template<typename T> void emplace(T node) { m_body.emplace_back(std::move(node)); }

	Register generate(size_t function_id, BytecodeGenerator &generator, ASTContext &) const final;

	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }

  private:
	void print_this_node(const std::string &indent) const override
	{
		spdlog::debug("{}Module", indent);
		spdlog::debug("{}  - body:", indent);
		std::string new_indent = indent + std::string(6, ' ');
		for (const auto &el : m_body) { el->print_node(new_indent); }
	}
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

	Register generate(size_t, BytecodeGenerator &, ASTContext &) const final;

	const std::shared_ptr<ASTNode> &test() const { return m_test; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &orelse() const { return m_orelse; }


  private:
	void print_this_node(const std::string &indent) const override
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

	Register generate(size_t, BytecodeGenerator &, ASTContext &) const final;

	const std::shared_ptr<ASTNode> &target() const { return m_target; }
	const std::shared_ptr<ASTNode> &iter() const { return m_iter; }
	const std::vector<std::shared_ptr<ASTNode>> &body() const { return m_body; }
	const std::vector<std::shared_ptr<ASTNode>> &orelse() const { return m_orelse; }
	const std::string &type_comment() const { return m_type_comment; }

  private:
	void print_this_node(const std::string &indent) const override
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

	Register generate(size_t, BytecodeGenerator &, ASTContext &) const final;

	const std::shared_ptr<ASTNode> &lhs() const { return m_lhs; }
	OpType op() const { return m_op; }
	const std::shared_ptr<ASTNode> &rhs() const { return m_rhs; }

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

	void print_this_node(const std::string &indent) const override
	{
		spdlog::debug("{}Compare", indent);
		std::string new_indent = indent + std::string(6, ' ');
		spdlog::debug("{}  - lhs:", indent);
		m_lhs->print_node(new_indent);
		spdlog::debug("{}  - op: {}", indent, op_type_to_string(m_op));
		spdlog::debug("{}  - rhs:", indent);
		m_rhs->print_node(new_indent);
	}
};

template<typename NodeType> std::shared_ptr<NodeType> as(std::shared_ptr<ASTNode> node);

template<typename FuncType> void ASTNode::visit(FuncType &&func) const
{
	switch (node_type()) {
#define __AST_NODE_TYPE(x) \
	case ASTNodeType::x:   \
		return func(static_cast<const x *>(this));
		AST_NODE_TYPES
#undef __AST_NODE_TYPE
	}
}
}// namespace ast