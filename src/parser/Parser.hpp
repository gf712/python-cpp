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
	std::shared_ptr<ast::Module> m_module;
	Lexer &m_lexer;
	size_t m_token_position{ 0 };

  public:
	struct CacheKey
	{
		const std::type_info &rule;
		Token token;
	};

	struct CacheHash
	{
		size_t operator()(const CacheKey &cache) const;
	};

	struct CacheEqual
	{
		bool operator()(const CacheKey &lhs, const CacheKey &rhs) const;
	};

	struct CacheValue
	{
		using ValueType = std::variant<std::shared_ptr<ast::ASTNode>, std::vector<Token>>;
		std::variant<bool, ValueType> value;
		size_t position;
	};

	std::unordered_map<CacheKey, std::optional<CacheValue>, CacheHash, CacheEqual> m_cache;

  public:
	Parser(Lexer &l) : m_module(std::make_shared<ast::Module>(l.filename())), m_lexer(l)
	{
		m_lexer.ignore_nl_token() = true;
		m_lexer.ignore_comments() = true;
	}

	Lexer &lexer() { return m_lexer; }

	std::shared_ptr<ast::Module> module() { return m_module; }

	const size_t &token_position() const { return m_token_position; }
	size_t &token_position() { return m_token_position; }

	void parse();
};

}// namespace parser