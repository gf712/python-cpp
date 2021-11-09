#pragma once

#include "utilities.hpp"

#include <deque>
#include <functional>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"

static constexpr size_t tab_size = 8;

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


#define ENUMERATE_TOKENS      \
	__TOKEN(EOF_)             \
	__TOKEN(ENDMARKER)        \
	__TOKEN(COMMENT)          \
	__TOKEN(NAME)             \
	__TOKEN(NUMBER)           \
	__TOKEN(STRING)           \
	__TOKEN(NEWLINE)          \
	__TOKEN(NL)               \
	__TOKEN(INDENT)           \
	__TOKEN(DEDENT)           \
	__TOKEN(OP)               \
	__TOKEN(LPAREN)           \
	__TOKEN(RPAREN)           \
	__TOKEN(LSQB)             \
	__TOKEN(RSQB)             \
	__TOKEN(COLON)            \
	__TOKEN(COMMA)            \
	__TOKEN(SEMI)             \
	__TOKEN(PLUS)             \
	__TOKEN(MINUS)            \
	__TOKEN(STAR)             \
	__TOKEN(SLASH)            \
	__TOKEN(VBAR)             \
	__TOKEN(AMPER)            \
	__TOKEN(LESS)             \
	__TOKEN(GREATER)          \
	__TOKEN(EQUAL)            \
	__TOKEN(DOT)              \
	__TOKEN(PERCENT)          \
	__TOKEN(LBRACE)           \
	__TOKEN(RBRACE)           \
	__TOKEN(EQEQUAL)          \
	__TOKEN(NOTEQUAL)         \
	__TOKEN(LESSEQUAL)        \
	__TOKEN(GREATEREQUAL)     \
	__TOKEN(TILDE)            \
	__TOKEN(CIRCUMFLEX)       \
	__TOKEN(LEFTSHIFT)        \
	__TOKEN(RIGHTSHIFT)       \
	__TOKEN(DOUBLESTAR)       \
	__TOKEN(PLUSEQUAL)        \
	__TOKEN(MINEQUAL)         \
	__TOKEN(STAREQUAL)        \
	__TOKEN(SLASHEQUAL)       \
	__TOKEN(PERCENTEQUAL)     \
	__TOKEN(AMPEREQUAL)       \
	__TOKEN(VBAREQUAL)        \
	__TOKEN(CIRCUMFLEXEQUAL)  \
	__TOKEN(LEFTSHIFTEQUAL)   \
	__TOKEN(RIGHTSHIFTEQUAL)  \
	__TOKEN(DOUBLESTAREQUAL)  \
	__TOKEN(DOUBLESLASH)      \
	__TOKEN(DOUBLESLASHEQUAL) \
	__TOKEN(AT)               \
	__TOKEN(ATEQUAL)          \
	__TOKEN(RARROW)           \
	__TOKEN(ELLIPSIS)         \
	__TOKEN(COLONEQUAL)

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
		: m_token_type(token_type), m_token_exact_type(token_type), m_start(start), m_end(end)
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

	static std::string_view stringify_token_type(const TokenType token_type)
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
};

#undef ENUMERATE_TOKENS

std::optional<std::string> read_file(const std::string &filename);

class Lexer
{

	std::deque<Token> m_tokens_to_emit;
	const std::string m_program;
	size_t m_cursor{ 0 };
	Position m_position;
	std::vector<size_t> m_indent_values;
	bool m_ignore_nl_token{ false };
	bool m_ignore_comments{ false };
	const std::string m_filename;

  private:
	Lexer(const Lexer &) = delete;
	Lexer(Lexer &&) = delete;
	Lexer &operator=(const Lexer &) = delete;
	Lexer &operator=(Lexer &&) = delete;

	Lexer(std::string &&program, std::string &&filename)
		: m_program(std::move(program)), m_position({ 0, 0, &m_program[0] }),
		  m_indent_values({ 0 }), m_filename(std::move(filename))
	{}

  public:
	static Lexer create(std::string program, std::string filename)
	{
		return Lexer(std::move(program), std::move(filename));
	}

	static Lexer create(std::string filename)
	{
		auto program = read_file(filename);
		if (!program.has_value()) { std::abort(); }
		auto &&p = std::move(program).value();
		return Lexer(std::move(p), std::move(filename));
	}

	const std::string &filename() const { return m_filename; }

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

	bool &ignore_nl_token() { return m_ignore_nl_token; }
	bool &ignore_comments() { return m_ignore_comments; }

  private:
	bool read_more_tokens()
	{
		if (m_cursor == m_program.size()) {
			auto original_position = m_position;
			increment_row_position();
			while (m_indent_values.size() > 1) {
				m_tokens_to_emit.emplace_back(Token::TokenType::DEDENT, m_position, m_position);
				m_indent_values.pop_back();
			}
			m_tokens_to_emit.emplace_back(
				Token::TokenType::ENDMARKER, original_position, m_position);
			return true;
		}

		if (try_empty_line()) { return true; }
		if (try_read_indent()) { return true; }
		if (try_read_newline()) { return true; }
		try_read_space();
		if (try_read_comment()) { return true; }
		if (try_read_name()) { return true; }
		if (try_read_string()) { return true; }
		if (try_read_operation()) { return true; }
		if (try_read_number()) { return true; }

		return false;
	}

	bool try_read_comment()
	{
		if (peek(0) == '#') {
			const auto original_position = m_position;
			advance(1);
			advance_while([](const char c) {
				// TODO: make newline check platform agnostic
				//       in windows this can be "\r\n"
				return std::isalnum(c) || c != '\n';
			});

			if (!m_ignore_comments) {
				m_tokens_to_emit.emplace_back(
					Token::TokenType::COMMENT, original_position, m_position);
			} else {
				increment_row_position();
				const auto current_position = m_position;
				// the comment is essentially removed from the source
				// which means a line with only a comment becomes NL
				// otherwise it's just a NEWLINE
				if (original_position.column == 0 && m_ignore_nl_token) {
					if (m_cursor > m_program.size()) {
						m_tokens_to_emit.emplace_back(
							Token::TokenType::NEWLINE, current_position, m_position);
					}
				} else if (!m_ignore_nl_token) {
					m_tokens_to_emit.emplace_back(
						Token::TokenType::NL, original_position, m_position);
					m_tokens_to_emit.emplace_back(
						Token::TokenType::NEWLINE, current_position, m_position);
				} else {
					m_tokens_to_emit.emplace_back(
						Token::TokenType::NEWLINE, current_position, m_position);
				}
			}

			return true;
		}

		return false;
	}

	bool try_empty_line()
	{
		const auto original_position = m_position;
		const auto cursor = m_cursor;
		// if it's a newline
		if (std::isspace(peek(0)) && !std::isblank(peek(0))) {
			if (original_position.column == 0) {
				increment_row_position();
				if (!m_ignore_nl_token) {
					m_tokens_to_emit.emplace_back(
						Token::TokenType::NL, original_position, m_position);
				}
				return true;
			} else {
				return false;
			}
		}
		while (advance_if([](const char c) { return std::isblank(c); })) {}
		if (std::isspace(peek(0))) {
			increment_row_position();
			if (!m_ignore_nl_token) {
				m_tokens_to_emit.emplace_back(Token::TokenType::NL, original_position, m_position);
			}
			return true;
		}
		m_position = original_position;
		m_cursor = cursor;
		return false;
	}

	bool try_read_indent()
	{
		// if we are the start of a new line we need to check indent/dedent
		if (m_position.column == 0) {
			const Position original_position = m_position;
			const auto [indent_value, position_increment] = compute_indent_level();
			increment_column_position(position_increment);
			if (indent_value > m_indent_values.back()) {
				m_tokens_to_emit.emplace_back(
					Token::TokenType::INDENT, original_position, m_position);
				m_indent_values.push_back(indent_value);
				return true;
			} else if (indent_value < m_indent_values.back()) {
				while (indent_value != m_indent_values.back()) {
					m_tokens_to_emit.emplace_back(
						Token::TokenType::DEDENT, original_position, m_position);
					m_indent_values.pop_back();
				}
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
		if (std::isalnum(peek(0))) return {};

		if (peek(0) == '(') return Token::TokenType::LPAREN;
		if (peek(0) == ')') return Token::TokenType::RPAREN;
		if (peek(0) == '[') return Token::TokenType::LSQB;
		if (peek(0) == ']') return Token::TokenType::RSQB;
		if (peek(0) == ':') return Token::TokenType::COLON;
		if (peek(0) == ',') return Token::TokenType::COMMA;
		if (peek(0) == ';') return Token::TokenType::SEMI;
		if (peek(0) == '+') return Token::TokenType::PLUS;
		if (peek(0) == '-') return Token::TokenType::MINUS;
		if (peek(0) == '*') return Token::TokenType::STAR;
		if (peek(0) == '/') return Token::TokenType::SLASH;
		if (peek(0) == '|') return Token::TokenType::VBAR;
		if (peek(0) == '&') return Token::TokenType::AMPER;
		if (peek(0) == '<') return Token::TokenType::LESS;
		if (peek(0) == '>') return Token::TokenType::GREATER;
		if (peek(0) == '=') return Token::TokenType::EQUAL;
		if (peek(0) == '.') return Token::TokenType::DOT;
		if (peek(0) == '%') return Token::TokenType::PERCENT;
		if (peek(0) == '{') return Token::TokenType::LBRACE;
		if (peek(0) == '}') return Token::TokenType::RBRACE;
		if (peek(0) == '~') return Token::TokenType::TILDE;
		if (peek(0) == '^') return Token::TokenType::CIRCUMFLEX;
		if (peek(0) == '@') return Token::TokenType::AT;
		return {};
	}

	std::optional<Token::TokenType> try_read_operation_with_two_characters() const
	{
		if (m_cursor + 1 >= m_program.size()) return {};
		if (std::isalnum(peek(0)) || std::isalnum(peek(1))) return {};
		if (peek(0) == '=' && peek(1) == '=') return Token::TokenType::EQEQUAL;
		if (peek(0) == '!' && peek(1) == '=') return Token::TokenType::NOTEQUAL;
		if (peek(0) == '<' && peek(1) == '=') return Token::TokenType::LESSEQUAL;
		if (peek(0) == '>' && peek(1) == '=') return Token::TokenType::GREATEREQUAL;
		if (peek(0) == '<' && peek(1) == '<') return Token::TokenType::LEFTSHIFT;
		if (peek(0) == '>' && peek(1) == '>') return Token::TokenType::RIGHTSHIFT;
		if (peek(0) == '*' && peek(1) == '*') return Token::TokenType::DOUBLESTAR;
		if (peek(0) == '+' && peek(1) == '=') return Token::TokenType::PLUSEQUAL;
		if (peek(0) == '-' && peek(1) == '=') return Token::TokenType::MINEQUAL;
		if (peek(0) == '*' && peek(1) == '=') return Token::TokenType::STAREQUAL;
		if (peek(0) == '%' && peek(1) == '=') return Token::TokenType::PERCENTEQUAL;
		if (peek(0) == '&' && peek(1) == '=') return Token::TokenType::AMPEREQUAL;
		if (peek(0) == '|' && peek(1) == '=') return Token::TokenType::VBAREQUAL;
		if (peek(0) == '^' && peek(1) == '=') return Token::TokenType::CIRCUMFLEXEQUAL;
		if (peek(0) == '/' && peek(1) == '/') return Token::TokenType::DOUBLESLASH;
		if (peek(0) == ':' && peek(1) == '=') return Token::TokenType::COLONEQUAL;
		if (peek(0) == '@' && peek(1) == '=') return Token::TokenType::ATEQUAL;
		if (peek(0) == '-' && peek(1) == '>') return Token::TokenType::RARROW;
		return {};
	}

	std::optional<Token::TokenType> try_read_operation_with_three_characters() const
	{
		if (m_cursor + 2 >= m_program.size()) return {};
		if (std::isalnum(peek(0)) || std::isalnum(peek(1)) || std::isalnum(peek(2))) return {};
		if (peek(0) == '<' && peek(1) == '<' && peek(2) == '=')
			return Token::TokenType::LEFTSHIFTEQUAL;
		if (peek(0) == '>' && peek(1) == '>' && peek(2) == '=')
			return Token::TokenType::RIGHTSHIFTEQUAL;
		if (peek(0) == '*' && peek(1) == '*' && peek(2) == '=')
			return Token::TokenType::DOUBLESTAREQUAL;
		if (peek(0) == '/' && peek(1) == '/' && peek(2) == '=')
			return Token::TokenType::DOUBLESLASHEQUAL;
		return {};
	}

	bool try_read_operation()
	{
		const Position original_position = m_position;
		std::optional<Token::TokenType> type;

		if (type = try_read_operation_with_three_characters(); type.has_value()) {
			advance(3);
		} else if (type = try_read_operation_with_two_characters(); type.has_value()) {
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
		auto valid_start_name = [](const char c) { return std::isalpha(c) || c == '_'; };

		if (!valid_start_name(peek(0))) { return false; }
		const Position original_position = m_position;
		// name must start with alpha
		if (!advance_if([valid_start_name](const char c) { return valid_start_name(c); })) {
			return false;
		}

		auto valid_name = [](const char c) { return std::isalnum(c) || c == '_'; };

		advance_while([valid_name](const char c) { return valid_name(c); });
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
		if (!condition(m_program[m_cursor])) { return false; }
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