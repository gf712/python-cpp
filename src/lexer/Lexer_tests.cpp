#include "Lexer.hpp"
#include "gtest/gtest.h"

namespace {
std::vector<Token> generate_tokens(std::string_view program)
{
	Lexer lexer{ std::string(program) };
	std::vector<Token> tokens;
	size_t i = 0;
	while (auto token = lexer.peek_token(i++)) { tokens.push_back(std::move(*token)); }
	return tokens;
}

void assert_generates_tokens(std::string_view program,
	const std::vector<Token::TokenType> &expected_tokens)
{
	const auto tokens = generate_tokens(program);
	ASSERT_EQ(expected_tokens.size(), tokens.size());
	for (size_t i = 0; i < expected_tokens.size(); ++i) {
		ASSERT_EQ(expected_tokens[i], tokens[i].token_type())
			<< fmt::format("Expected {}, but got {}",
				   Token::stringify_token_type(expected_tokens[i]),
				   Token::stringify_token_type(tokens[i].token_type()));
	}
}
}// namespace

TEST(Lexer, SimpleNumericAssignment)
{
	constexpr std::string_view program = "a = 2\n";
	std::vector<Token::TokenType> expected_tokens{ Token::TokenType::NAME,
		Token::TokenType::EQUAL,
		Token::TokenType::NUMBER,
		Token::TokenType::NEWLINE,
		Token::TokenType::ENDMARKER };
	assert_generates_tokens(program, expected_tokens);
}

TEST(Lexer, SimpleStringAssignment)
{
	constexpr std::string_view program = "a = \"2\"\n";
	std::vector<Token::TokenType> expected_tokens{ Token::TokenType::NAME,
		Token::TokenType::EQUAL,
		Token::TokenType::STRING,
		Token::TokenType::NEWLINE,
		Token::TokenType::ENDMARKER };
	assert_generates_tokens(program, expected_tokens);
}

TEST(Lexer, BinaryOperationWithAssignment)
{
	constexpr std::string_view program = "a = 1 + 1\n";
	std::vector<Token::TokenType> expected_tokens{ Token::TokenType::NAME,
		Token::TokenType::EQUAL,
		Token::TokenType::NUMBER,
		Token::TokenType::PLUS,
		Token::TokenType::NUMBER,
		Token::TokenType::NEWLINE,
		Token::TokenType::ENDMARKER };
	assert_generates_tokens(program, expected_tokens);
}

TEST(Lexer, FunctionDefinition)
{
	constexpr std::string_view program =
		"def add(a, b):\n"
		"   return a + b\n";
	std::vector<Token::TokenType> expected_tokens{ Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::LPAREN,
		Token::TokenType::NAME,
		Token::TokenType::COMMA,
		Token::TokenType::NAME,
		Token::TokenType::RPAREN,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::PLUS,
		Token::TokenType::NAME,
		Token::TokenType::NEWLINE,
		Token::TokenType::DEDENT,
		Token::TokenType::ENDMARKER };
	assert_generates_tokens(program, expected_tokens);
}

TEST(Lexer, FunctionDefinitionAndCall)
{
	constexpr std::string_view program =
		"def add(a, b):\n"
		"   return a + b\n"
		"c = add(1, 1)\n";
	std::vector<Token::TokenType> expected_tokens{ Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::LPAREN,
		Token::TokenType::NAME,
		Token::TokenType::COMMA,
		Token::TokenType::NAME,
		Token::TokenType::RPAREN,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::PLUS,
		Token::TokenType::NAME,
		Token::TokenType::NEWLINE,
		Token::TokenType::DEDENT,
		Token::TokenType::NAME,
		Token::TokenType::EQUAL,
		Token::TokenType::NAME,
		Token::TokenType::LPAREN,
		Token::TokenType::NUMBER,
		Token::TokenType::COMMA,
		Token::TokenType::NUMBER,
		Token::TokenType::RPAREN,
		Token::TokenType::NEWLINE,
		Token::TokenType::ENDMARKER };
	assert_generates_tokens(program, expected_tokens);
}

TEST(Lexer, DoubleIndentationAtProgramEnd)
{
	constexpr std::string_view program =
		"if foo:\n"
		"  if bar:\n"
		"    print(42)\n";
	std::vector<Token::TokenType> expected_tokens{
		Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::LPAREN,
		Token::TokenType::NUMBER,
		Token::TokenType::RPAREN,
		Token::TokenType::NEWLINE,
		Token::TokenType::DEDENT,
		Token::TokenType::DEDENT,
		Token::TokenType::ENDMARKER,
	};
	assert_generates_tokens(program, expected_tokens);
}

TEST(Lexer, DoubleNestedIndentationAtProgramEnd)
{
	constexpr std::string_view program =
		"if foo:\n"
		"  if bar:\n"
		"    print(42)\n"
		"  else:\n"
		"    print(43)\n";
	std::vector<Token::TokenType> expected_tokens{
		Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::LPAREN,
		Token::TokenType::NUMBER,
		Token::TokenType::RPAREN,
		Token::TokenType::NEWLINE,
		Token::TokenType::DEDENT,
		Token::TokenType::NAME,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::LPAREN,
		Token::TokenType::NUMBER,
		Token::TokenType::RPAREN,
		Token::TokenType::NEWLINE,
		Token::TokenType::DEDENT,
		Token::TokenType::DEDENT,
		Token::TokenType::ENDMARKER,
	};
	assert_generates_tokens(program, expected_tokens);
}

TEST(Lexer, IfStatementEqualityCheck)
{
	constexpr std::string_view program =
		"a = 1\n"
		"if a == 1:\n"
		"  a = 2\n"
		"else:\n"
		"  a = 3\n";
	std::vector<Token::TokenType> expected_tokens{
		Token::TokenType::NAME,
		Token::TokenType::EQUAL,
		Token::TokenType::NUMBER,
		Token::TokenType::NEWLINE,
		Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::EQEQUAL,
		Token::TokenType::NUMBER,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::EQUAL,
		Token::TokenType::NUMBER,
		Token::TokenType::NEWLINE,
		Token::TokenType::DEDENT,
		Token::TokenType::NAME,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::EQUAL,
		Token::TokenType::NUMBER,
		Token::TokenType::NEWLINE,
		Token::TokenType::DEDENT,
		Token::TokenType::ENDMARKER,
	};
	assert_generates_tokens(program, expected_tokens);
}


TEST(Lexer, FunctionWithUnderscoreDefinition)
{
	constexpr std::string_view program =
		"def plus_one(a):\n"
		"   constant = 1\n"
		"   return a + constant\n";
	std::vector<Token::TokenType> expected_tokens{ Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::LPAREN,
		Token::TokenType::NAME,
		Token::TokenType::RPAREN,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::EQUAL,
		Token::TokenType::NUMBER,
		Token::TokenType::NEWLINE,
		Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::PLUS,
		Token::TokenType::NAME,
		Token::TokenType::NEWLINE,
		Token::TokenType::DEDENT,
		Token::TokenType::ENDMARKER };
	assert_generates_tokens(program, expected_tokens);
}

TEST(Lexer, FunctionWithIfElseReturn)
{
	constexpr std::string_view program =
		"def foo(a):\n"
		"	if a == 1:\n"
		"		return 10\n"
		"	else:\n"
		"		return 2\n"
		"a = foo(1)\n";
	std::vector<Token::TokenType> expected_tokens{ Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::LPAREN,
		Token::TokenType::NAME,
		Token::TokenType::RPAREN,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::NAME,
		Token::TokenType::EQEQUAL,
		Token::TokenType::NUMBER,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::NUMBER,
		Token::TokenType::NEWLINE,
		Token::TokenType::DEDENT,
		Token::TokenType::NAME,
		Token::TokenType::COLON,
		Token::TokenType::NEWLINE,
		Token::TokenType::INDENT,
		Token::TokenType::NAME,
		Token::TokenType::NUMBER,
		Token::TokenType::NEWLINE,
		Token::TokenType::DEDENT,
		Token::TokenType::DEDENT,
		Token::TokenType::NAME,
		Token::TokenType::EQUAL,
		Token::TokenType::NAME,
		Token::TokenType::LPAREN,
		Token::TokenType::NUMBER,
		Token::TokenType::RPAREN,
		Token::TokenType::NEWLINE,
		Token::TokenType::ENDMARKER };
	assert_generates_tokens(program, expected_tokens);
}


TEST(Lexer, LiteralList)
{
	constexpr std::string_view program = "a = [1, 2, 3, 5]\n";

	std::vector<Token::TokenType> expected_tokens{ Token::TokenType::NAME,
		Token::TokenType::EQUAL,
		Token::TokenType::LSQB,
		Token::TokenType::NUMBER,
		Token::TokenType::COMMA,
		Token::TokenType::NUMBER,
		Token::TokenType::COMMA,
		Token::TokenType::NUMBER,
		Token::TokenType::COMMA,
		Token::TokenType::NUMBER,
		Token::TokenType::RSQB,
		Token::TokenType::NEWLINE,
		Token::TokenType::ENDMARKER };
	assert_generates_tokens(program, expected_tokens);
}