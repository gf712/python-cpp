#pragma once

#include "utilities.hpp"

#include <string_view>
#include <iostream>
#include <optional>
#include <vector>
#include <functional>
#include <deque>

#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"

static constexpr size_t tab_size = 4;

struct Position
{
	size_t row;
	size_t column;
	const char *pointer_to_program;

	friend std::ostream &operator<<(std::ostream &os, const Position &position)
	{
		os << position.row + 1 << ':' << position.column;
		return os;
	}
};


template<> struct fmt::formatter<Position>
{
	constexpr auto parse(format_parse_context &ctx) { return ctx.end(); }

	template<typename FormatContext> auto format(const Position &pos, FormatContext &ctx)
	{
		return format_to(ctx.out(), "{}:{}", pos.row + 1, pos.column);
	}
};


#define ENUMERATE_TOKENS \
	__TOKEN(EOF_)        \
	__TOKEN(ENDMARKER)   \
	__TOKEN(NAME)        \
	__TOKEN(OP)          \
	__TOKEN(INDENT)      \
	__TOKEN(DEDENT)      \
	__TOKEN(NEWLINE)     \
	__TOKEN(NUMBER)      \
	__TOKEN(RPAREN)      \
	__TOKEN(LPAREN)      \
	__TOKEN(COLON)       \
	__TOKEN(EQUAL)       \
	__TOKEN(PLUS)        \
	__TOKEN(MINUS)       \
	__TOKEN(STAR)        \
	__TOKEN(DOUBLESTAR)  \
	__TOKEN(SLASH)       \
	__TOKEN(LEFTSHIFT)   \
	__TOKEN(RIGHTSHIFT)  \
	__TOKEN(LSQB)        \
	__TOKEN(RSQB)        \
	__TOKEN(SEMI)        \
	__TOKEN(COMMA)       \
	__TOKEN(COLONEQUAL)  \
	__TOKEN(STRING)

class Token
{
  public:
	enum class TokenType;

  private:
	const TokenType m_token_type;
	const TokenType m_token_exact_type;
	const Position m_start;
	const Position m_end;

  public:
	enum class TokenType {
#define __TOKEN(x) x,
		ENUMERATE_TOKENS
#undef __TOKEN
	};

	Token(TokenType token_type, const Position start, const Position end)
		: m_token_type(token_type), m_token_exact_type(exact_type_from_token(token_type)),
		  m_start(start), m_end(end)
	{}

	TokenType token_type() const { return m_token_type; }

	Position start() const { return m_start; }

	Position end() const { return m_end; }

	std::string to_string() const
	{
		std::string value{ m_start.pointer_to_program, m_end.pointer_to_program };
		if (m_token_exact_type == TokenType::NEWLINE) { value = "\\n"; }
		return fmt::format("{:<12} {:>1}:{:>3} - {:>1}:{:>3} \t\"{}\"",
			stringify_token_type(m_token_type),
			m_start.row,
			m_start.column,
			m_end.row,
			m_end.column,
			value);
	}

	friend std::ostream &operator<<(std::ostream &os, const Token &t)
	{
		return os << t.to_string();
	}

  private:
	std::string_view stringify_token_type(const TokenType token_type) const
	{
		switch (token_type) {

#define __TOKEN(x)     \
	case TokenType::x: \
		return #x;
			ENUMERATE_TOKENS
#undef __TOKEN
		}
		ASSERT_NOT_REACHED()
	}

	TokenType exact_type_from_token(TokenType token) { return token; }
};

#undef ENUMERATE_TOKENS

class Lexer
{

	std::deque<Token> m_tokens_to_emit;
	const std::string m_program;
	size_t m_cursor{ 0 };
	Position m_position;
	size_t m_current_indent_level{ 0 };
	size_t m_current_indent_value{ 0 };

  private:
	Lexer(const Lexer &) = delete;
	Lexer(Lexer &&) = delete;
	Lexer &operator=(const Lexer &) = delete;
	Lexer &operator=(Lexer &&) = delete;

  public:
	Lexer(std::string program) : m_program(std::move(program)), m_position({ 0, 0, &m_program[0] })
	{}

	std::optional<Token> next_token()
	{
		if (!m_tokens_to_emit.empty()) return pop_front();
		if (m_tokens_to_emit.empty() && m_cursor > m_program.size()) { return {}; }
		if (read_more_tokens()) { return pop_front(); }
		if (m_tokens_to_emit.empty() && m_cursor != m_program.size()) {
			spdlog::error("Failed to parse program at position {}", m_position);
			std::abort();
		}
		ASSERT_NOT_REACHED();
	}

	std::optional<Token> peek_token(size_t positions)
	{
		while (positions >= m_tokens_to_emit.size()) {
			if (m_cursor > m_program.size()) { return {}; }
			if (!read_more_tokens()) { return {}; }
		}
		ASSERT(positions < m_tokens_to_emit.size())
		return m_tokens_to_emit[positions];
	}

	std::string_view get(Position start, Position end)
	{
		const auto size = static_cast<size_t>(end.pointer_to_program - start.pointer_to_program);
		return std::string_view{ start.pointer_to_program, size };
	}

  private:
	bool read_more_tokens()
	{
		if (m_cursor == m_program.size()) {
			auto original_position = m_position;
			increment_row_position();
			while (m_current_indent_level > 0) {
				m_tokens_to_emit.emplace_back(Token::TokenType::DEDENT, m_position, m_position);
				m_current_indent_level--;
			}
			m_tokens_to_emit.emplace_back(
				Token::TokenType::ENDMARKER, original_position, m_position);
			return true;
		}

		if (try_read_indent()) { return true; }
		if (try_read_newline()) { return true; }
		try_read_space();
		if (try_read_name()) { return true; }
		if (try_read_string()) { return true; }
		if (try_read_operation()) { return true; }
		if (try_read_number()) { return true; }

		return false;
	}

	bool try_read_indent()
	{
		// if we are the start of a new line we need to check indent/dedent
		if (m_position.column == 0) {
			const Position original_position = m_position;
			const auto [indent_value, position_increment] = compute_indent_level();
			increment_column_position(position_increment);
			if (indent_value > m_current_indent_value) {
				m_tokens_to_emit.emplace_back(
					Token::TokenType::INDENT, original_position, m_position);
				m_current_indent_level++;
			} else if (indent_value < m_current_indent_value) {
				m_tokens_to_emit.emplace_back(
					Token::TokenType::DEDENT, original_position, m_position);
				m_current_indent_level--;
			}
			if (m_current_indent_value != indent_value) {
				m_current_indent_value = indent_value;
				return true;
			}
		}
		return false;
	}

	std::tuple<size_t, size_t> compute_indent_level()
	{
		size_t pos = 0;
		size_t indentation_value = 0;
		while (std::isblank(peek(pos))) {
			if (peek(pos) == '\t') {
				indentation_value += tab_size;
			} else {
				indentation_value++;
			}
			pos += 1;
		}
		return { indentation_value, pos };
	}

	bool try_read_number()
	{
		const Position original_position = m_position;

		if (!(peek(0) == '.' || std::isdigit(peek(0)))) { return false; }

		bool is_decimal_part = peek(0) == '.';
		size_t position = 1;

		while (std::isdigit(peek(position)) || (!is_decimal_part && peek(position) == '.')) {
			position++;
		}
		advance(position);
		m_tokens_to_emit.emplace_back(Token::TokenType::NUMBER, original_position, m_position);

		return true;
	}

	bool try_read_string()
	{
		const Position original_position = m_position;

		if (!(peek(0) == '\"' || peek(0) == '\'')) { return false; }

		const bool double_quote = peek(0) == '\"';

		size_t position = 1;

		if (double_quote) {
			while (!(peek(position) == '\"')) { position++; }
		} else {
			while (!(peek(position) == '\'')) { position++; }
		}
		advance(position + 1);

		m_tokens_to_emit.emplace_back(Token::TokenType::STRING, original_position, m_position);

		return true;
	}

	std::optional<Token::TokenType> try_read_operation_with_one_character() const
	{
		if (peek(0) == '(')
			return Token::TokenType::LPAREN;
		else if (peek(0) == ')')
			return Token::TokenType::RPAREN;
		else if (peek(0) == ':')
			return Token::TokenType::COLON;
		else if (peek(0) == '=')
			return Token::TokenType::EQUAL;
		else if (peek(0) == '+')
			return Token::TokenType::PLUS;
		else if (peek(0) == '-')
			return Token::TokenType::MINUS;
		else if (peek(0) == '*')
			return Token::TokenType::STAR;
		else if (peek(0) == '/')
			return Token::TokenType::SLASH;
		else if (peek(0) == ',')
			return Token::TokenType::COMMA;
		return {};
	}

	std::optional<Token::TokenType> try_read_operation_with_two_characters() const
	{
		if (peek(0) == '*' && peek(1) == '*') return Token::TokenType::DOUBLESTAR;
		if (peek(0) == '<' && peek(1) == '<') return Token::TokenType::LEFTSHIFT;
		if (peek(0) == '>' && peek(1) == '>') return Token::TokenType::RIGHTSHIFT;
		if (peek(0) == ':' && peek(1) == '=') return Token::TokenType::COLONEQUAL;

		return {};
	}

	bool try_read_operation()
	{
		const Position original_position = m_position;
		std::optional<Token::TokenType> type;

		if (type = try_read_operation_with_two_characters(); type.has_value()) {
			advance(2);
		} else if (type = try_read_operation_with_one_character(); type.has_value()) {
			advance(1);
		} else {
			return false;
		}

		m_tokens_to_emit.emplace_back(*type, original_position, m_position);

		return true;
	}

	bool try_read_space()
	{
		const size_t whitespace_size = advance_while([](const char c) { return std::isblank(c); });
		if (whitespace_size > 0) {
			return true;
		} else {
			return false;
		}
	}

	bool try_read_newline()
	{
		if (!peek("\n")) { return false; }

		const Position original_position = m_position;
		increment_row_position();
		m_tokens_to_emit.emplace_back(Token::TokenType::NEWLINE, original_position, m_position);
		return true;
	}

	bool try_read_name()
	{
		if (!std::isalpha(peek(0))) { return false; }
		const Position original_position = m_position;
		// name must start with alpha
		if (!advance_if([](const char c) { return std::isalpha(c); })) { return false; }
		advance_while([](const char c) { return std::isalnum(c) || c == '_'; });
		m_tokens_to_emit.emplace_back(Token::TokenType::NAME, original_position, m_position);
		return true;
	}

	char peek(size_t i) const
	{
		ASSERT(m_cursor + i < m_program.size())
		return m_program[m_cursor + i];
	}

	bool peek(std::string_view pattern) const
	{
		int j = 0;
		for (size_t i = m_cursor; i < m_cursor + pattern.size(); ++i) {
			if (m_program[i] != pattern[j++]) { return false; }
		}
		return true;
	}

	template<typename ConditionType> size_t advance_while(ConditionType &&condition)
	{
		const size_t start = m_cursor;
		while (condition(m_program[m_cursor])) { increment_column_position(1); }
		return m_cursor - start;
	}

	void advance(std::string_view pattern) { advance(pattern.size()); }

	void advance(const size_t positions)
	{
		m_cursor += positions;
		m_position.column += positions;
		m_position.pointer_to_program = &m_program[m_cursor];
	}

	bool advance_if(const char pattern)
	{
		if (m_program[m_cursor] == pattern) {
			increment_column_position(1);
			return true;
		}
		return false;
	}

	template<typename ConditionType> bool advance_if(ConditionType &&condition)
	{
		if (!condition(m_program[m_cursor])) {
			return false;
		}
		increment_column_position(1);
		return true;
	}

	Token pop_front()
	{
		auto &result = m_tokens_to_emit.front();
		m_tokens_to_emit.pop_front();
		return result;
	}

	void increment_row_position()
	{
		m_cursor++;
		m_position.column = 0;
		m_position.row++;
		m_position.pointer_to_program = &m_program[m_cursor];
	}

	void increment_column_position(size_t idx)
	{
		m_cursor += idx;
		m_position.column += idx;
		m_position.pointer_to_program = &m_program[m_cursor];
	}
};