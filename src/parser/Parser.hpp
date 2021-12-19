#pragma once

#include "ast/AST.hpp"
#include "lexer/Lexer.hpp"
#include "utilities.hpp"

#include <memory>
#include <string_view>
#include <tuple>
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
		while (m_token_position > 0) {
			m_lexer.next_token();
			m_token_position--;
		}
	}

	void parse();
};

class BlockScope
{
	Parser &m_p;

  public:
	BlockScope(Parser &p) : m_p(p) { m_p.m_stack.emplace_back(); }

	~BlockScope() { m_p.m_stack.pop_back(); }

	std::deque<std::shared_ptr<ast::ASTNode>> &parent()
	{
		ASSERT(m_p.m_stack.size() > 1)
		return m_p.m_stack[m_p.m_stack.size() - 2];
	}
};

}// namespace parser