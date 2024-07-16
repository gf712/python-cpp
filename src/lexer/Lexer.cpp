#include "Lexer.hpp"

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

std::optional<std::string> read_file(const std::string &filename)
{
	std::filesystem::path path = filename;
	if (!std::filesystem::exists(path)) {
		std::cerr << fmt::format("File {} does not exist", path.c_str()) << std::endl;
		return {};
	}

	std::ifstream in(std::filesystem::absolute(path).c_str());
	if (!in.is_open()) {
		std::cerr << fmt::format("Failed to open {}", std::filesystem::absolute(path).c_str())
				  << std::endl;
		return {};
	}

	std::string program;

	in.seekg(0, std::ios::end);
	program.reserve(in.tellg());
	in.seekg(0, std::ios::beg);

	program.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	if (program.back() != '\n') { program.append("\n"); }

	spdlog::debug("Input program: \n----\n{}\n----\n", program.c_str());
	return program;
}

void Lexer::push_token(Token::TokenType token_type, const Position &start, const Position &end)
{
	m_tokens_to_emit.emplace_back(token_type, start, end);
	m_current_line_tokens.emplace_back(token_type, start, end);
}

bool Lexer::read_more_tokens_loop()
{
	if (m_cursor > m_program.size()) { return false; }
	if (m_cursor == m_program.size()) {
		auto original_position = m_position;
		increment_row_position();
		m_cursor++;
		while (m_indent_values.size() > 1) {
			push_token(Token::TokenType::DEDENT, m_position, m_position);
			m_indent_values.pop_back();
		}
		push_token(Token::TokenType::ENDMARKER, original_position, m_position);
		return true;
	}

	if (current_mode() == Mode::NORMAL) {
		if (try_empty_line()) { return true; }
		if (try_read_comment()) { return true; }
		if (try_read_indent()) { return true; }
		if (try_read_newline()) { return true; }
		try_read_backslash();
		try_read_space();
		if (try_read_string()) { return true; }
		if (try_read_name()) { return true; }
		if (try_read_operation()) { return true; }
		if (try_read_number()) { return true; }
	} else {
		if (try_fstring()) { return true; }
	}

	return false;
}

bool Lexer::read_more_tokens()
{
	const size_t token_size = m_tokens_to_emit.size();
	while (token_size >= m_tokens_to_emit.size()) {
		bool read_tokens = read_more_tokens_loop();
		if (read_tokens) {
			std::erase_if(m_tokens_to_emit, [this](Token &token) {
				if (m_ignore_comments && token.token_type() == Token::TokenType::COMMENT) {
					return true;
				}
				if (m_ignore_nl_token && token.token_type() == Token::TokenType::NL) {
					return true;
				}
				return false;
			});
		} else {
			return false;
		}
	}
	return true;
}

void Lexer::push_new_line()
{
	const auto original_position = m_position;
	increment_column_position(1);
	if (m_parenthesis_level > 0) {
		push_token(Token::TokenType::NL, original_position, m_position);
	} else {
		push_token(Token::TokenType::NEWLINE, original_position, m_position);
	}
	increment_row_position();
}

void Lexer::push_empty_line()
{
	const auto original_position = m_position;
	increment_column_position(1);
	push_token(Token::TokenType::NL, original_position, m_position);
	increment_row_position();
}


std::optional<size_t> Lexer::comment_start() const
{
	size_t n = 0;
	if (m_cursor >= m_program.size()) { return {}; }
	if (peek(n) == '#') { return n; }
	while (std::isblank(peek(n)) && peek(n) != '\n') {
		if (peek(n + 1) == '#') { return n + 1; }
		n++;
	}
	return {};
}

bool Lexer::try_read_comment()
{
	if (auto start = comment_start(); start.has_value()) {
		// advance to '#'
		advance(*start);
		const auto original_position = m_position;

		// skip '#'
		advance(1);
		advance_while([](const char c) {
			// TODO: make newline check platform agnostic in Windows this can be "\r\n"
			return std::isalnum(c) || c != '\n';
		});

		push_token(Token::TokenType::COMMENT, original_position, m_position);
		// if we want NL tokens, we need to look back and see if this
		// line has a non-comment token
		// for example:
		// import foo # my import
		// would be tokenized to NAME NAME COMMENT NEWLINE
		// but
		// import foo
		// # comment
		// would be tokenized to NAME NAME NEWLINE COMMENT NL
		// this is important to distiguish a no op line such as COMMENT NL
		// from a end of line ENDLINE
		if (m_current_line_tokens.size() == 1) {
			// just a comment in this line
			ASSERT(m_current_line_tokens.back().token_type() == Token::TokenType::COMMENT)
			push_empty_line();
		} else {
			push_new_line();
		}
		return true;
	}

	return false;
}

bool Lexer::try_empty_line()
{
	const auto original_position = m_position;
	const auto cursor = m_cursor;
	// if it's a newline
	if (std::isspace(peek(0)) && !std::isblank(peek(0))) {
		if (original_position.column == 0) {
			push_empty_line();
			return true;
		} else {
			return false;
		}
	}
	while (advance_if([](const char c) { return std::isblank(c); })) {}
	if (m_cursor >= m_program.size()) { return false; }
	if (std::isspace(peek(0))) {
		push_empty_line();
		return true;
	}
	m_position = original_position;
	m_cursor = cursor;
	return false;
}

bool Lexer::try_read_indent()
{
	// if we are the start of a new line we need to check indent/dedent
	// and we are not inside of parenthesis
	if (m_position.column == 0 && m_parenthesis_level == 0) {
		const Position original_position = m_position;
		const auto [indent_value, position_increment] = compute_indent_level();
		increment_column_position(position_increment);
		if (indent_value > m_indent_values.back()) {
			push_token(Token::TokenType::INDENT, original_position, m_position);
			m_indent_values.push_back(indent_value);
			return true;
		} else if (indent_value < m_indent_values.back()) {
			while (indent_value != m_indent_values.back()) {
				push_token(Token::TokenType::DEDENT, original_position, m_position);
				m_indent_values.pop_back();
			}
			return true;
		}
	}
	return false;
}

std::tuple<size_t, size_t> Lexer::compute_indent_level()
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

template<typename FunctorType>
std::optional<size_t> Lexer::parse_digits(size_t n, FunctorType &&f) const
{
	bool previous_is_underscore = false;
	while (f(peek(n)) || peek(n) == '_') {
		if (peek(n) == '_' && previous_is_underscore) {
			// invalid decimal
			// TODO: Python interpreter should throw syntax error
			return {};
		} else if (peek(n) == '_') {
			previous_is_underscore = true;
		} else {
			previous_is_underscore = false;
		}
		n++;
	}
	if (n > 0 && peek(n - 1) == '_') {
		// number cannot end in an underscore
		TODO();
		return {};
	}
	return n;
}

std::optional<size_t> Lexer::int_number(size_t n) const
{
	// ugly, but it isn't trivial to chain/pipe these operations in a generic way
	if (auto end = hex_number(n)) {
		return *end;
	} else if (auto end = bin_number(n)) {
		return *end;
	} else if (auto end = oct_number(n)) {
		return *end;
	} else if (auto end = dec_number(n)) {
		return *end;
	} else {
		return {};
	}
}

std::optional<size_t> Lexer::hex_number(size_t n) const
{
	if (peek(n++) != '0') { return {}; }
	if (peek(n) != 'x' && peek(n) != 'X') { return {}; }
	n++;
	auto is_hex_value = [](const unsigned char val) {
		// [0-9a-fA-F]
		return std::isdigit(val) || (val >= 'a' && val <= 'f') || (val >= 'A' && val <= 'F');
	};
	if (!is_hex_value(peek(n))) { return {}; }
	n++;
	auto result = parse_digits(n, is_hex_value);
	if (result.has_value() && *result < 3) { return {}; }
	return result;
}

std::optional<size_t> Lexer::bin_number(size_t n) const
{
	if (peek(n++) != '0') { return {}; }
	if (peek(n) != 'b' && peek(n) != 'B') { return {}; }
	n++;
	auto is_bin_value = [](const unsigned char val) {
		// [01]
		return val == '0' || val == '1';
	};
	if (!is_bin_value(peek(n))) { return {}; }
	n++;
	auto result = parse_digits(n, is_bin_value);
	if (result.has_value() && *result < 3) { return {}; }
	return result;
}

std::optional<size_t> Lexer::oct_number(size_t n) const
{
	if (peek(n++) != '0') { return {}; }
	if (peek(n) != 'o' && peek(n) != 'O') { return {}; }
	n++;
	auto is_oct_value = [](const unsigned char val) {
		// [0-7]
		return val >= '0' && val <= '7';
	};
	if (!is_oct_value(peek(n))) { return {}; }
	n++;
	auto result = parse_digits(n, is_oct_value);
	if (result.has_value() && *result < 3) { return {}; }
	return result;
}

std::optional<size_t> Lexer::dec_number(size_t n) const
{
	bool previous_is_underscore = false;
	// (?:0(?:_?0)*
	if (peek(n) == '_') {
		// number cannot start with underscore
		TODO();
		return {};
	}
	auto result = parse_digits(n, [](const char c) { return c == '0'; });
	if (!result) { return {}; }
	n = *result;
	if (n > 0) {
		// we found a valid decimal number with just 0s (possibly with '_')
		return n;
	}

	ASSERT(n == 0)
	ASSERT(previous_is_underscore == false)
	// [1-9](?:_?[0-9])*)
	if (!std::isdigit(peek(n))) { return {}; }
	n++;
	result = parse_digits(n, [](const char c) { return std::isdigit(c); });
	if (!result) { return {}; }
	n = *result;
	if (n > 0) { return n; }
	return {};
}

std::optional<size_t> Lexer::exp_number(size_t n) const
{
	if (peek(n) != 'e' && peek(n) != 'E') { return {}; }
	n++;
	if (peek(n) == '-' || peek(n) == '+') { n++; }

	if (!std::isdigit(peek(n))) { return {}; }
	n++;
	return parse_digits(n, [](const char c) { return std::isdigit(c); });
}

std::optional<size_t> Lexer::point_float_number(size_t n) const
{
	// [0-9](?:_?[0-9])*
	if (std::isdigit(peek(n))) {
		n++;
		auto result = parse_digits(n, [](const char c) { return std::isdigit(c); });
		if (!result) { return {}; }
		n = *result;
	}

	// \.(?:[0-9](?:_?[0-9])*)?`
	if (peek(n) != '.') { return {}; }
	n++;
	if (!std::isdigit(peek(n))) { return n; }
	n++;
	return parse_digits(n, [](const char c) { return std::isdigit(c); });
}

std::optional<size_t> Lexer::exp_float_number(size_t n) const
{
	// [0-9](?:_?[0-9])*)
	if (!(peek(n) >= '0' && peek(n) <= '9')) { return {}; }
	n++;
	auto result = parse_digits(n, [](const char c) { return std::isdigit(c); });
	if (result) { return exp_number(*result); }
	return {};
}

std::optional<size_t> Lexer::imag_number(size_t n) const
{
	auto result = [this](size_t n) -> std::optional<size_t> {
		// [0-9](?:_?[0-9])*)
		if (!(peek(n) >= '0' && peek(n) <= '9')) { return {}; }
		n++;
		auto result = parse_digits(n, [](const char c) { return std::isdigit(c); });
		if (!result) { return {}; }
		n = *result;

		if (!(peek(n) == 'j' && peek(n) == 'J')) { return {}; }
		n++;
		return n;
	}(n);

	if (!result) {
		result = [this](size_t n) -> std::optional<size_t> {
			if (auto result = float_number(n)) {
				if (peek(*result) == 'j' || peek(*result) == 'J') { return *result + 1; }
			}
			return {};
		}(n);
	}

	return result;
}

std::optional<size_t> Lexer::float_number(size_t n) const
{
	if (auto end = point_float_number(n)) {
		return *end;
	} else if (auto end = exp_float_number(n)) {
		return *end;
	} else {
		return {};
	}
}

bool Lexer::try_read_number()
{
	const Position original_position = m_position;
	size_t n = 0;

	if (m_cursor >= m_program.size()) { return false; }

	if (auto end = imag_number(n)) {
		increment_column_position(*end);
	} else if (auto end = float_number(n)) {
		increment_column_position(*end);
	} else if (auto end = int_number(n)) {
		increment_column_position(*end);
	} else {
		return false;
	}
	push_token(Token::TokenType::NUMBER, original_position, m_position);
	return true;
}

bool Lexer::single_quote_string(size_t n)
{
	const auto original_position = m_position;
	const auto original_cursor = m_cursor;
	while (true) {
		// [^'\\]*(?:\\.[^'\\]*)*'

		// [^'\\]*
		while (peek(n) != '\\' && peek(n) != '\'') { n++; }

		auto advance = [this](size_t n) -> std::optional<size_t> {
			// (?:\\.[^'\\]*)
			if (peek(n++) != '\\') return {};
			if (peek(n++) == '\n') return {};
			while (peek(n) != '\\' && peek(n) != '\'') { n++; }
			return n;
		};

		// (?:\\.[^'\\]*)*
		while (true) {
			if (auto inc = advance(n); inc.has_value()) {
				n = *inc;
			} else {
				break;
			}
		}

		if (peek(n) == '\\' && peek(n + 1) == '\n') {
			Lexer::advance(n + 2);
			n = 0;
			increment_row_position();
			continue;
		}
		// '
		if (peek(n) != '\'') {
			m_position = original_position;
			m_cursor = original_cursor;
			return false;
		}
		Lexer::advance(n + 1);
		n = 0;
		break;
	}

	return true;
}

bool Lexer::double_quote_string(size_t n)
{
	const auto original_position = m_position;
	const auto original_cursor = m_cursor;
	while (true) {
		// [^"\\]*(?:\\.[^"\\]*)*"

		// [^"\\]*
		while (peek(n) != '\\' && peek(n) != '\"') { n++; }

		auto advance = [this](size_t n) -> std::optional<size_t> {
			// (?:\\.[^"\\]*)
			if (peek(n++) != '\\') return {};
			if (peek(n++) == '\n') return {};
			while (peek(n) != '\\' && peek(n) != '\"') { n++; }
			return n;
		};

		// (?:\\.[^"\\]*)*
		while (true) {
			if (auto inc = advance(n); inc.has_value()) {
				n = *inc;
			} else {
				break;
			}
		}

		if (peek(n) == '\\' && peek(n + 1) == '\n') {
			Lexer::advance(n + 2);
			n = 0;
			increment_row_position();
			continue;
		}

		// "
		if (peek(n) != '"') {
			m_position = original_position;
			m_cursor = original_cursor;
			return false;
		}
		Lexer::advance(n + 1);
		n = 0;
		break;
	}

	return true;
}


bool Lexer::single_triple_quote_string(size_t n)
{
	const auto original_position = m_position;
	const auto original_cursor = m_cursor;
	while (true) {
		// [^'\\]*(?:(?:\\.|'(?!''))[^'\\]*)*'''

		// [^'\\]*
		while (peek(n) != '\\' && peek(n) != '\'') {
			if (peek(n) == '\n') { increment_row_position(); }
			n++;
		}

		auto advance = [this](size_t n) -> std::optional<size_t> {
			// (?:\\.|'(?!''))

			if (peek(n) == '\\' && peek(n + 1) != '\n') {
				// \\.
				return n + 2;
			} else {
				// '(?!'')

				// '
				if (peek(n) != '\'') return {};
				if (peek(n) == '\n') { increment_row_position(); }
				n++;

				// (?!'')
				if (peek(n) == '\'' && peek(n + 1) == '\'') return {};

				return n;
			}
		};

		// (?:(?:\\.|'(?!''))[^'\\]*)*
		while (true) {
			if (auto inc = advance(n); inc.has_value()) {
				n = *inc;
				while (peek(n) != '\\' && peek(n) != '\'') {
					if (peek(n) == '\n') { increment_row_position(); }
					n++;
				}
			} else {
				break;
			}
		}

		if (peek(n) == '\\' && peek(n + 1) == '\n') {
			Lexer::advance(n + 2);
			n = 0;
			increment_row_position();
			continue;
		}

		// '''
		if (peek(n) != '\'' || peek(n + 1) != '\'' || peek(n + 2) != '\'') {
			m_position = original_position;
			m_cursor = original_cursor;
			return false;
		}
		Lexer::advance(n + 3);
		n = 0;
		break;
	}

	return true;
}

bool Lexer::double_triple_quote_string(size_t n)
{
	const auto original_position = m_position;
	const auto original_cursor = m_cursor;
	while (true) {
		// [^"\\]*(?:(?:\\.|"(?!""))[^"\\]*)*"""

		// [^"\\]*
		while (peek(n) != '\\' && peek(n) != '\"') {
			if (peek(n) == '\n') { increment_row_position(); }
			n++;
		}

		auto advance = [this](size_t n) -> std::optional<size_t> {
			// (?:\\.|"(?!""))

			if (peek(n) == '\\' && peek(n + 1) != '\n') {
				// \\.
				return n + 2;
			} else {
				// "(?!"")

				// "
				if (peek(n) != '\"') return {};
				if (peek(n) == '\n') { increment_row_position(); }
				n++;

				// (?!"")
				if (peek(n) == '\"' && peek(n + 1) == '\"') return {};

				return n;
			}
		};

		// (?:(?:\\.|"(?!""))[^"\\]*)*
		while (true) {
			if (auto inc = advance(n); inc.has_value()) {
				n = *inc;
				while (peek(n) != '\\' && peek(n) != '\"') {
					if (peek(n) == '\n') { increment_row_position(); }
					n++;
				}
			} else {
				break;
			}
		}

		if (peek(n) == '\\' && peek(n + 1) == '\n') {
			Lexer::advance(n + 2);
			n = 0;
			increment_row_position();
			continue;
		}

		// """
		if (peek(n) != '\"' || peek(n + 1) != '\"' || peek(n + 2) != '\"') {
			m_position = original_position;
			m_cursor = original_cursor;
			return false;
		}
		Lexer::advance(n + 3);
		n = 0;
		break;
	}

	return true;
}

bool Lexer::try_read_string()
{
	if (m_cursor >= m_program.size()) { return false; }

	static constexpr std::array<std::string_view, 25> string_possible_prefixes = { "",
		"br",
		"f",
		"rB",
		"u",
		"rF",
		"U",
		"fr",
		"RF",
		"fR",
		"BR",
		"RB",
		"bR",
		"rb",
		"F",
		"rf",
		"Fr",
		"R",
		"b",
		"Br",
		"B",
		"r",
		"FR",
		"Rf",
		"Rb" };

	const Position original_position = m_position;
	const auto original_cursor = m_cursor;

	size_t idx = 0;

	const auto it = std::find_if(string_possible_prefixes.begin(),
		string_possible_prefixes.end(),
		[this, idx](std::string_view el) {
			for (size_t i = idx; i < el.size(); ++i) {
				if (peek(i) != el[i]) return false;
			}
			return peek(idx + el.size()) == '\'' || peek(idx + el.size()) == '\"';
		});

	if (it == string_possible_prefixes.end()) { return false; }

	idx += it->size();

	auto is_triple_quote = [this, idx](const char quote) {
		ASSERT(quote == '\'' || quote == '\"');
		return peek(idx) == quote && peek(idx + 1) == quote && peek(idx + 2) == quote;
	};

	const bool is_fstring =
		(it->find('f') != std::string_view::npos || it->find('F') != std::string_view::npos);

	auto push_string_token =
		[this, original_position, is_fstring, original_cursor, is_triple_quote, idx]() {
			if (!is_fstring) {
				push_token(Token::TokenType::STRING, original_position, m_position);
			} else {
				m_cursor = original_cursor;
				m_position = original_position;

				auto quote_size = (is_triple_quote('\'') || is_triple_quote('\"')) ? 3 : 1;
				m_quote.push(quote_type(std::string_view{ m_program.begin() + m_cursor + idx,
					m_program.begin() + m_cursor + idx + quote_size }));

				Lexer::advance(idx + quote_size);
				push_token(Token::TokenType::FSTRING_START, original_position, m_position);
				m_mode.push(Mode::FSTRING);
				m_fstring_paren_level.push(m_parenthesis_level);
			}
		};

	if (is_triple_quote('\'')) {
		idx += 3;
		if (single_triple_quote_string(idx)) {
			push_string_token();
			return true;
		} else {
			return false;
		}
	} else if (is_triple_quote('\"')) {
		idx += 3;
		if (double_triple_quote_string(idx)) {
			push_string_token();
			return true;
		} else {
			return false;
		}
	} else if (peek(idx) == '\'') {
		idx += 1;
		if (single_quote_string(idx)) {
			push_string_token();
			return true;
		} else {
			return false;
		}
	} else if (peek(idx) == '\"') {
		idx += 1;
		if (double_quote_string(idx)) {
			push_string_token();
			return true;
		} else {
			return false;
		}
	}
	return false;
}

bool Lexer::try_fstring()
{
	const auto original_position = m_position;

	auto reached_end = [this, original_position]() {
		auto position = m_position;
		if ((m_cursor + 1) < m_program.size() && peek(0) == '}') {
			if (peek(1) != '}') {
				push_token(Token::TokenType::FSTRING_MIDDLE, original_position, position);
				advance(1);
				push_token(Token::TokenType::RBRACE, position, m_position);
				return true;
			} else {
				advance(2);
				return false;
			}
		} else if ((peek(0) == '\'' && m_quote.top() == Quote::SINGLE_SINGLE_QUOTE)
				   || (peek(0) == '"' && m_quote.top() == Quote::SINGLE_DOUBLE_QUOTE)) {
			push_token(Token::TokenType::FSTRING_MIDDLE, original_position, position);
			advance(1);
			push_token(Token::TokenType::FSTRING_END, position, m_position);
			m_mode.pop();
			m_quote.pop();
			m_fstring_paren_level.pop();
			return true;
		} else if (((peek(0) == '\'' && peek(1) == '\'' && peek(2) == '\'')
					   && m_quote.top() == Quote::TRIPLE_SINGLE_QUOTE)
				   || ((peek(0) == '\"' && peek(1) == '\"' && peek(2) == '\"')
					   && m_quote.top() == Quote::TRIPLE_DOUBLE_QUOTE)) {
			push_token(Token::TokenType::FSTRING_MIDDLE, original_position, position);
			advance(3);
			push_token(Token::TokenType::FSTRING_END, position, m_position);
			m_mode.pop();
			m_quote.pop();
			m_fstring_paren_level.pop();
			return true;
		} else if (peek(0) == '{') {
			if (peek(1) != '{') {
				push_token(Token::TokenType::FSTRING_MIDDLE, original_position, position);
				advance(1);
				push_token(Token::TokenType::LBRACE, position, m_position);
				m_mode.push(Mode::NORMAL);
				return true;
			} else {
				advance(2);
				return false;
			}
		} else {
			advance(1);
		}
		return false;
	};

	while (!reached_end()) {
		ASSERT(!peek("\n"));
	}

	return true;
}


std::optional<Token::TokenType> Lexer::try_read_operation_with_one_character()
{
	if (std::isalnum(peek(0))) return {};

	if (peek(0) == '(') {
		m_parenthesis_level++;
		return Token::TokenType::LPAREN;
	}
	if (peek(0) == ')') {
		ASSERT(m_parenthesis_level > 0);
		m_parenthesis_level--;
		return Token::TokenType::RPAREN;
	}
	if (peek(0) == '[') {
		m_parenthesis_level++;
		return Token::TokenType::LSQB;
	}
	if (peek(0) == ']') {
		ASSERT(m_parenthesis_level > 0);
		m_parenthesis_level--;
		return Token::TokenType::RSQB;
	}
	if (peek(0) == '{') {
		m_parenthesis_level++;
		return Token::TokenType::LBRACE;
	}
	if (peek(0) == '}') {
		if (m_mode.size() > 1) {
			ASSERT(m_mode.top() == Mode::NORMAL);
			m_mode.pop();
			ASSERT(m_mode.top() == Mode::FSTRING);
		} else {
			ASSERT(m_parenthesis_level > 0);
			m_parenthesis_level--;
		}
		return Token::TokenType::RBRACE;
	}
	if (peek(0) == ':') {
		if (m_mode.size() > 1 && m_parenthesis_level == m_fstring_paren_level.top()) {
			ASSERT(m_mode.top() == Mode::NORMAL);
			m_mode.pop();
			ASSERT(m_mode.top() == Mode::FSTRING);
		}
		return Token::TokenType::COLON;
	}
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
	if (peek(0) == '~') return Token::TokenType::TILDE;
	if (peek(0) == '^') return Token::TokenType::CIRCUMFLEX;
	if (peek(0) == '@') return Token::TokenType::AT;
	if (peek(0) == '!')
		return Token::TokenType::EXCLAMATION;// only appears in Python 3.11, but makes fstring
											 // easier
	return {};
}

std::optional<Token::TokenType> Lexer::try_read_operation_with_two_characters() const
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
	if (peek(0) == '/' && peek(1) == '=') return Token::TokenType::SLASHEQUAL;
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

std::optional<Token::TokenType> Lexer::try_read_operation_with_three_characters() const
{
	if (m_cursor + 2 >= m_program.size()) return {};
	if (std::isalnum(peek(0)) || std::isalnum(peek(1)) || std::isalnum(peek(2))) return {};
	if (peek(0) == '<' && peek(1) == '<' && peek(2) == '=') return Token::TokenType::LEFTSHIFTEQUAL;
	if (peek(0) == '>' && peek(1) == '>' && peek(2) == '=')
		return Token::TokenType::RIGHTSHIFTEQUAL;
	if (peek(0) == '*' && peek(1) == '*' && peek(2) == '=')
		return Token::TokenType::DOUBLESTAREQUAL;
	if (peek(0) == '/' && peek(1) == '/' && peek(2) == '=')
		return Token::TokenType::DOUBLESLASHEQUAL;
	if (peek(0) == '.' && peek(1) == '.' && peek(2) == '.') return Token::TokenType::ELLIPSIS;
	return {};
}

bool Lexer::try_read_operation()
{
	const Position original_position = m_position;
	std::optional<Token::TokenType> type;

	if (m_cursor >= m_program.size()) { return false; }

	if (type = try_read_operation_with_three_characters(); type.has_value()) {
		advance(3);
	} else if (type = try_read_operation_with_two_characters(); type.has_value()) {
		advance(2);
	} else if (type = try_read_operation_with_one_character(); type.has_value()) {
		advance(1);
	} else {
		return false;
	}

	push_token(*type, original_position, m_position);

	return true;
}

void Lexer::try_read_backslash()
{
	try_read_space();
	if (advance_if('\\')) {
		if (advance_if('\n')) {
			increment_row_position();
		} else {
			// ERROR!
			// TODO: Emit an error token
			TODO();
		}
	}
}

bool Lexer::try_read_space()
{
	const size_t whitespace_size = advance_while([](const char c) { return std::isblank(c); });
	if (whitespace_size > 0) {
		return true;
	} else {
		return false;
	}
}

bool Lexer::try_read_newline()
{
	if (!peek("\n")) { return false; }
	push_new_line();
	return true;
}

bool Lexer::try_read_name()
{
	if (m_cursor >= m_program.size()) { return false; }

	auto valid_start_name = [](const char c) { return std::isalpha(c) || c == '_'; };

	if (!valid_start_name(peek(0))) { return false; }
	const Position original_position = m_position;
	// name must start with alpha
	if (!advance_if([valid_start_name](const char c) { return valid_start_name(c); })) {
		return false;
	}

	auto valid_name = [](const char c) { return std::isalnum(c) || c == '_'; };

	advance_while([valid_name](const char c) { return valid_name(c); });
	push_token(Token::TokenType::NAME, original_position, m_position);
	return true;
}

char Lexer::peek(size_t i) const
{
	ASSERT(m_cursor + i < m_program.size())
	return m_program[m_cursor + i];
}

bool Lexer::peek(std::string_view pattern) const
{
	int j = 0;
	for (size_t i = m_cursor; i < m_cursor + pattern.size(); ++i) {
		if (m_program[i] != pattern[j++]) { return false; }
	}
	return true;
}

void Lexer::increment_row_position()
{
	m_position.column = 0;
	m_position.row++;
	m_position.pointer_to_program = &m_program[m_cursor];
	m_current_line_tokens.clear();
}

void Lexer::increment_column_position(size_t idx)
{
	m_cursor += idx;
	m_position.column += idx;
	m_position.pointer_to_program = &m_program[m_cursor];
}

void Lexer::advance(const size_t positions)
{
	m_cursor += positions;
	m_position.column += positions;
	m_position.pointer_to_program = &m_program[m_cursor];
}

bool Lexer::advance_if(const char pattern)
{
	if (m_program[m_cursor] == pattern) {
		increment_column_position(1);
		return true;
	}
	return false;
}

Token Lexer::pop_front()
{
	auto &result = m_tokens_to_emit.front();
	m_tokens_to_emit.pop_front();
	return result;
}
