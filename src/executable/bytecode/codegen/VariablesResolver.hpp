#pragma once

#include "ast/AST.hpp"

#include <set>

class VariablesResolver : public ast::CodeGenerator
{
  public:
	enum class Visibility {
		GLOBAL = 0,
		NAME = 1,
		LOCAL = 2,
		CELL = 3,
		FREE = 4,
		HIDDEN = 5,// assignments in a class scope are hidden, ie. cannot be referred to
	};

	struct Symbol
	{
		std::string name;
		Visibility visibility;
		SourceLocation source_location;

		auto operator<=>(const Symbol &other) const = default;
	};

	struct SymbolMap
	{
		std::set<Symbol> symbols;
		bool contains(std::string name) const
		{
			return std::find_if(symbols.begin(), symbols.end(), [&name](const auto &s) {
				return s.name == name;
			}) != symbols.end();
		}

		std::optional<std::reference_wrapper<const Symbol>> get_hidden_symbol(
			std::string name) const
		{
			auto it = std::find_if(symbols.begin(), symbols.end(), [&name](const auto &s) {
				return s.name == name && s.visibility == Visibility::HIDDEN;
			});
			if (it == symbols.end()) { return std::nullopt; }
			return *it;
		}

		std::optional<std::reference_wrapper<const Symbol>> get_visible_symbol(
			std::string name) const
		{
			auto it = std::find_if(symbols.begin(), symbols.end(), [&name](const auto &s) {
				return s.name == name && s.visibility != Visibility::HIDDEN;
			});
			if (it == symbols.end()) { return std::nullopt; }
			return *it;
		}

		std::optional<std::reference_wrapper<const Symbol>> get_symbol(std::string name) const
		{
			auto it = std::find_if(
				symbols.begin(), symbols.end(), [&name](const auto &s) { return s.name == name; });
			if (it == symbols.end()) { return std::nullopt; }
			return *it;
		}

		void add_symbol(Symbol s)
		{
			if (s.visibility == Visibility::HIDDEN) {
				ASSERT(!get_hidden_symbol(s.name).has_value());
			} else {
				ASSERT(!get_visible_symbol(s.name).has_value());
			}
			symbols.insert(std::move(s));
		}
		void delete_symbol(const Symbol &s) { symbols.erase(s); }
	};

	struct Scope : NonCopyable
	{
		enum class Type { MODULE = 0, FUNCTION = 1, CLASS = 2, CLOSURE = 3 };

		std::string name;
		std::string namespace_;
		Type type;
		Scope *parent;
		std::vector<std::reference_wrapper<Scope>> children;
		std::set<std::string> captures;
		SymbolMap symbol_map;
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

	void store(const std::string &name, SourceLocation source_location, Scope::Type type);
	void load(const std::string &name, SourceLocation source_location, Scope::Type type);
	void annotate_free_and_cell_variables(const std::string &name);
	Scope *top_level_node(const std::string &name) const;
	Scope *find_outer_scope_of_type(Scope::Type) const;

	template<typename FunctionType> void visit_function(FunctionType *function);

  public:
	static VisibilityMap resolve(ast::Module *node)
	{
		auto resolver = VariablesResolver();
		node->codegen(&resolver);
		return std::move(resolver.m_visibility);
	}
};
