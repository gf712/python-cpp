#pragma once

#include "ast/AST.hpp"
#include "lexer/Lexer.hpp"
#include "bytecode/BytecodeGenerator.hpp"
#include "utilities.hpp"

#include <string_view>
#include <memory>
#include <variant>
#include <tuple>

namespace parser {

class Parser
{
	std::shared_ptr<ast::Module> m_module;
	std::deque<std::shared_ptr<ast::ASTNode>> m_stack;
	std::vector<std::shared_ptr<ast::ASTNode>> m_statements;
	Lexer &m_lexer;
	size_t m_token_position{ 0 };

  public:
	Parser(Lexer &l) : m_module(std::make_shared<ast::Module>()), m_lexer(l) {}

	void push_to_stack(std::shared_ptr<ast::ASTNode> node) { m_stack.push_back(std::move(node)); }

	void clear_stack() { m_stack.clear(); }

	std::shared_ptr<ast::ASTNode> pop_stack()
	{
		auto node = m_stack.back();
		m_stack.pop_back();
		return node;
	}

	void print_stack() const
	{
		size_t i = 0;
		for (const auto &el : m_stack) { el->print_node(fmt::format("stack {} -> ", i++)); }
	}

	Lexer &lexer() { return m_lexer; }

	std::shared_ptr<ast::Module> module() { return m_module; }
	const std::vector<std::shared_ptr<ast::ASTNode>> &statements() const { return m_statements; }
	std::vector<std::shared_ptr<ast::ASTNode>> &statements() { return m_statements; }

	const std::deque<std::shared_ptr<ast::ASTNode>> &stack() const { return m_stack; }

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
}// namespace parser