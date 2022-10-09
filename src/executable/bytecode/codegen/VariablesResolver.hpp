#pragma once

#include "ast/AST.hpp"

#include <set>

class VariablesResolver : public ast::CodeGenerator
{
  public:
	enum class Visibility { GLOBAL = 0, NAME = 1, LOCAL = 2, CELL = 3, FREE = 4 };

	struct Scope : NonCopyable
	{
		enum class Type { MODULE = 0, FUNCTION = 1, CLASS = 2, CLOSURE = 3 };

		std::string name;
		std::string namespace_;
		Type type;
		Scope *parent;
		std::vector<std::reference_wrapper<Scope>> children;
		std::set<std::string> captures;
		std::unordered_map<std::string, Visibility> visibility;
		// this is beyond the scope of this class, but it is simple and cheap to check if a function
		// is a generator at this stage
		bool is_generator{ false };
		bool requires_class_ref{ false };
	};

	using VisibilityMap = std::unordered_map<std::string, std::unique_ptr<Scope>>;

  private:
	VisibilityMap m_visibility;
	std::deque<std::pair<std::reference_wrapper<Scope>, std::shared_ptr<ast::ASTNode>>> m_to_visit;
	std::optional<std::reference_wrapper<Scope>> m_current_scope;

	VariablesResolver() = default;
#define __AST_NODE_TYPE(NodeType) ast::Value *visit(const ast::NodeType *node) override;
	AST_NODE_TYPES
#undef __AST_NODE_TYPE

	void store(const std::string &name, SourceLocation source_location);
	void load(const std::string &name, SourceLocation source_location);
	void annotate_free_and_cell_variables(const std::string &name);
	Scope *top_level_node(const std::string &name) const;

	template<typename FunctionType> void visit_function(FunctionType *function);

  public:
	static VisibilityMap resolve(ast::Module *node)
	{
		auto resolver = VariablesResolver();
		node->codegen(&resolver);
		return std::move(resolver.m_visibility);
	}
};