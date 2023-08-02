#pragma once

#include "utilities.hpp"

#include <deque>
#include <functional>
#include <iostream>
#include <optional>
#include <stack>
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
	auto operator<=>(const Position &other) const = default;
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
	__TOKEN(COLONEQUAL)       \
	__TOKEN(EXCLAMATION)      \
	__TOKEN(FSTRING_START)    \
	__TOKEN(FSTRING_MIDDLE)   \
	__TOKEN(FSTRING_END)

class Token
{
  public:
	enum class TokenType;

  private:
	TokenType m_token_type;
	TokenType m_token_exact_type;
	Position m_start;
	Position m_end;

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
		ASSERT_NOT_REACHED();
	}
};

#undef ENUMERATE_TOKENS

std::optional<std::string> read_file(const std::string &filename);

class Lexer
{
  public:
	enum class Mode {
		NORMAL,
		FSTRING,
	};

	enum class Quote {
		SINGLE_SINGLE_QUOTE,
		SINGLE_DOUBLE_QUOTE,
		TRIPLE_SINGLE_QUOTE,
		TRIPLE_DOUBLE_QUOTE,
	};

  private:
	std::deque<Token> m_tokens_to_emit;
	const std::string m_program;
	size_t m_cursor{ 0 };
	Position m_position;
	std::vector<size_t> m_indent_values;
	bool m_ignore_nl_token{ false };
	bool m_ignore_comments{ false };
	const std::string m_filename;
	std::vector<Token> m_current_line_tokens;
	size_t m_parenthesis_level{ 0 };
	std::stack<Mode> m_mode;
	std::stack<Quote> m_quote;
	std::stack<size_t> m_fstring_paren_level;

  private:
	Lexer(const Lexer &) = delete;
	Lexer(Lexer &&) = delete;
	Lexer &operator=(const Lexer &) = delete;
	Lexer &operator=(Lexer &&) = delete;

	Lexer(std::string &&program, std::string &&filename)
		: m_program(std::move(program)), m_position({ 0, 0, &m_program[0] }),
		  m_indent_values({ 0 }), m_filename(std::move(filename)), m_mode({ Mode::NORMAL })
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

	const std::string &program() const { return m_program; }

	const std::string &filename() const { return m_filename; }

	// std::optional<Token> next_token()
	// {
	// 	if (!m_tokens_to_emit.empty()) return pop_front();
	// 	if (read_more_tokens()) { return pop_front(); }
	// 	if (m_tokens_to_emit.empty() && m_cursor > m_program.size()) { return {}; }
	// 	if (m_tokens_to_emit.empty()) {
	// 		spdlog::error("Failed to parse program at position {}", m_position);
	// 		std::abort();
	// 	}
	// 	ASSERT_NOT_REACHED();
	// }

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
	void push_token(Token::TokenType token_type, const Position &start, const Position &end);

	bool read_more_tokens();

	bool read_more_tokens_loop();

	void push_new_line();

	void push_empty_line();

	std::optional<size_t> comment_start() const;

	void try_read_backslash();

	bool try_read_comment();

	bool try_empty_line();

	bool try_read_indent();

	std::tuple<size_t, size_t> compute_indent_level();

	bool try_read_number();

	bool try_read_string();

	std::optional<Token::TokenType> try_read_operation_with_one_character();

	std::optional<Token::TokenType> try_read_operation_with_two_characters() const;

	std::optional<Token::TokenType> try_read_operation_with_three_characters() const;

	bool try_read_operation();

	bool try_read_space();

	bool try_read_newline();

	bool try_read_name();

	bool try_fstring();

	std::optional<size_t> hex_number(size_t) const;
	std::optional<size_t> bin_number(size_t) const;
	std::optional<size_t> oct_number(size_t) const;
	std::optional<size_t> dec_number(size_t) const;
	std::optional<size_t> int_number(size_t) const;
	std::optional<size_t> exp_number(size_t) const;
	std::optional<size_t> point_float_number(size_t) const;
	std::optional<size_t> exp_float_number(size_t) const;
	std::optional<size_t> float_number(size_t) const;
	std::optional<size_t> imag_number(size_t) const;

	bool single_quote_string(size_t);
	bool double_quote_string(size_t);
	bool single_triple_quote_string(size_t);
	bool double_triple_quote_string(size_t);

	template<typename FunctorType>
	std::optional<size_t> parse_digits(size_t n, FunctorType &&f) const;

	char peek(size_t i) const;

	bool peek(std::string_view pattern) const;

	template<typename ConditionType> size_t advance_while(ConditionType &&condition)
	{
		const size_t start = m_cursor;
		while (condition(m_program[m_cursor])) { increment_column_position(1); }
		return m_cursor - start;
	}

	void advance(std::string_view pattern) { advance(pattern.size()); }

	void advance(const size_t positions);

	bool advance_if(const char pattern);

	template<typename ConditionType> bool advance_if(ConditionType &&condition)
	{
		if (!condition(m_program[m_cursor])) { return false; }
		increment_column_position(1);
		return true;
	}

	Token pop_front();

	void increment_row_position();

	void increment_column_position(size_t idx);

	Mode current_mode() const { return m_mode.top(); }

	Quote quote_type(std::string_view str)
	{
		if (str == "'") {
			return Quote::SINGLE_SINGLE_QUOTE;
		} else if (str == "\"") {
			return Quote::SINGLE_DOUBLE_QUOTE;
		} else if (str == "\'\'\'") {
			return Quote::TRIPLE_SINGLE_QUOTE;
		} else if (str == "\"\"\"") {
			return Quote::TRIPLE_DOUBLE_QUOTE;
		}
		ASSERT_NOT_REACHED();
	}
};
