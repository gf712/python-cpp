#pragma once

#include "ast/AST.hpp"
#include "lexer/Lexer.hpp"
#include "utilities.hpp"

#include <memory>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <variant>

namespace parser {

class Parser
{
	friend class BlockScope;

	std::shared_ptr<ast::Module> m_module;
	std::deque<std::deque<std::shared_ptr<ast::ASTNode>>> m_stack;
	Lexer &m_lexer;
	size_t m_token_position{ 0 };

  public:
	struct CacheLine
	{
		std::tuple<std::array<void *, 10>, size_t> type_matcher_ids;
		Token token;
	};

	struct CacheHash
	{
		size_t operator()(const CacheLine &cache) const;
	};

	struct CacheEqual
	{
		bool operator()(const CacheLine &lhs, const CacheLine &rhs) const;
	};

	std::unordered_set<CacheLine, CacheHash, CacheEqual> m_cache;

  public:
	Parser(Lexer &l) : m_module(std::make_shared<ast::Module>(l.filename())), m_lexer(l)
	{
		m_lexer.ignore_nl_token() = true;
		m_lexer.ignore_comments() = true;

		m_stack.emplace_back();
	}

	void push_to_stack(std::shared_ptr<ast::ASTNode> node)
	{
		m_stack.back().push_back(std::move(node));
	}

	void clear_stack() { m_stack.clear(); }

	std::shared_ptr<ast::ASTNode> pop_back()
	{
		ASSERT(!m_stack.empty())
		ASSERT(!m_stack.back().empty())
		auto node = m_stack.back().back();
		m_stack.back().pop_back();
		return node;
	}

	std::shared_ptr<ast::ASTNode> pop_front()
	{
		ASSERT(!m_stack.empty())
		ASSERT(!m_stack.back().empty())
		auto node = m_stack.back().front();
		m_stack.back().pop_front();
		return node;
	}

	void print_stack() const
	{
		size_t i = 0;
		for (const auto &block : m_stack) {
			for (const auto &el : block) { el->print_node(fmt::format("stack {} -> ", i)); }
			i++;
		}
	}

	Lexer &lexer() { return m_lexer; }

	std::shared_ptr<ast::Module> module() { return m_module; }

	const std::deque<std::shared_ptr<ast::ASTNode>> &stack() const { return m_stack.back(); }

	const size_t &token_position() const { return m_token_position; }
	size_t &token_position() { return m_token_position; }

	void commit()
	{
		// while (m_token_position > 0) {
		// 	m_lexer.next_token();
		// 	m_token_position--;
		// }
	}

	void parse();
};

class BlockScope
{
	Parser &m_p;
	size_t m_level;

  public:
	BlockScope(Parser &p) : m_p(p), m_level(m_p.m_stack.size() - 1) { m_p.m_stack.emplace_back(); }

	~BlockScope()
	{
		while (m_p.m_stack.size() > (m_level + 1)) { m_p.m_stack.pop_back(); }
	}

	std::deque<std::shared_ptr<ast::ASTNode>> &parent()
	{
		ASSERT(m_p.m_stack.size() > m_level)
		return m_p.m_stack[m_level];
	}
};

}// namespace parser