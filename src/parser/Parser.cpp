#include "Parser.hpp"

#define PARSER_ERROR()                                       \
	spdlog::error("Parser error {}:{}", __FILE__, __LINE__); \
	std::abort();

using namespace ast;
using namespace parser;

template<typename Derived> struct Pattern
{
	virtual ~Pattern() = default;

	static bool matches(Parser &p)
	{
		const auto start_stack_size = p.stack().size();
		const auto start_position = p.token_position();
		const bool is_match = Derived::matches_impl(p);
		if (!is_match) {
			while (p.stack().size() > start_stack_size) { p.pop_stack(); }
			p.token_position() = start_position;
		}
		return is_match;
	}
};

template<size_t TypeIdx, typename PatternTuple> class PatternMatch_
{
	template<typename T, typename = void> struct has_advance_by : std::false_type
	{
	};

	template<typename T>
	struct has_advance_by<T, decltype(std::declval<T>().advance_by, void())> : std::true_type
	{
	};

  public:
	PatternMatch_() {}
	static bool match(Parser &p)
	{
		using CurrentType = typename std::tuple_element<TypeIdx, PatternTuple>::type;
		const size_t original_token_position = p.token_position();
		if (CurrentType::matches(p)) {
			// std::cout << "match:    " << *p.lexer().peek_token(original_token_position) << '\n';
			if constexpr (has_advance_by<CurrentType>::value) {
				p.token_position() += CurrentType::advance_by;
			}
			if constexpr (TypeIdx + 1 == std::tuple_size_v<PatternTuple>) {
				return true;
			} else {
				return PatternMatch_<TypeIdx + 1, PatternTuple>::match(p);
			}
		} else {
			// std::cout << "no match: " << *p.lexer().peek_token(original_token_position) << '\n';
			p.token_position() = original_token_position;
			return false;
		}
	}
};

template<typename... PatternType> class PatternMatch
{
  public:
	PatternMatch() {}
	static bool match(Parser &p)
	{
		const auto start_token_position = p.token_position();
		const auto start_stack_size = p.stack().size();
		const bool is_match = PatternMatch_<0, std::tuple<PatternType...>>::match(p);
		if (!is_match) {
			while (p.stack().size() > start_stack_size) { p.pop_stack(); }
			p.token_position() = start_token_position;
		}
		return is_match;
	}
};


template<Token::TokenType... Rest> struct ComposedTypes
{
};

template<Token::TokenType Head, Token::TokenType... Rest> struct ComposedTypes<Head, Rest...>
{
	static constexpr size_t size = 1 + sizeof...(Rest);
	static constexpr Token::TokenType head = Head;
	using tail = ComposedTypes<Rest...>;
};

template<Token::TokenType Head> struct ComposedTypes<Head>
{
	static constexpr size_t size = 1;
	static constexpr Token::TokenType head = Head;
};

template<typename... PatternTypes>
struct OneOrMorePattern : Pattern<OneOrMorePattern<PatternTypes...>>
{
	static bool matches_impl(Parser &p)
	{
		using PatternType = PatternMatch<PatternTypes...>;
		if (!PatternType::match(p)) { return false; }
		auto original_token_position = p.token_position();
		while (PatternType::match(p)) { original_token_position = p.token_position(); }
		p.token_position() = original_token_position;
		return true;
	}
};


// ApplyInBetweenPattern<NamedExpressionPattern, SingleTokenPattern<Token::TokenType::COMMA>>>
template<typename MainPatternType, typename InBetweenPattern>
struct ApplyInBetweenPattern : Pattern<ApplyInBetweenPattern<MainPatternType, InBetweenPattern>>
{
	static bool matches_impl(Parser &p)
	{
		using MainPatternType_ = PatternMatch<MainPatternType>;
		using InBetweenPattern_ = PatternMatch<InBetweenPattern>;

		if (!MainPatternType_::match(p)) { return false; }

		while (InBetweenPattern_::match(p)) {
			if (!MainPatternType_::match(p)) { break; }
		}
		spdlog::debug(
			"ApplyInBetweenPattern: {}", p.lexer().peek_token(p.token_position())->to_string());

		return true;
	}
};


template<typename... PatternTypes>
struct ZeroOrMorePattern : Pattern<ZeroOrMorePattern<PatternTypes...>>
{
	static bool matches_impl(Parser &p)
	{
		using PatternType = PatternMatch<PatternTypes...>;
		if (!PatternType::match(p)) { return true; }
		auto original_token_position = p.token_position();
		while (PatternType::match(p)) { original_token_position = p.token_position(); }
		p.token_position() = original_token_position;
		return true;
	}
};

template<typename... PatternTypes>
struct ZeroOrOnePattern : Pattern<ZeroOrOnePattern<PatternTypes...>>
{
	static bool matches_impl(Parser &p)
	{
		using PatternType = PatternMatch<PatternTypes...>;
		auto original_token_position = p.token_position();
		if (PatternType::match(p)) { return true; }
		p.token_position() = original_token_position;
		spdlog::debug("ZeroOrOnePattern (no match): {}",
			p.lexer().peek_token(p.token_position())->to_string());
		return true;
	}
};


template<typename PatternsType> struct SingleTokenPattern_
{
	static bool match(Parser &p)
	{
		if (PatternsType::head == p.lexer().peek_token(p.token_position())->token_type()) {
			return true;
		}
		if constexpr (PatternsType::size == 1) {
			return false;
		} else {
			if (SingleTokenPattern_<typename PatternsType::tail>::match(p)) {
				p.token_position()++;
				return true;
			} else {
				return false;
			}
		}
	}
};

template<Token::TokenType... Patterns>
struct SingleTokenPattern : Pattern<SingleTokenPattern<Patterns...>>
{
	static constexpr size_t advance_by = 1;

	static bool matches_impl(Parser &p)
	{
		return SingleTokenPattern_<ComposedTypes<Patterns...>>::match(p);
	}
};


template<typename PatternType> struct NegativeLookAhead : Pattern<NegativeLookAhead<PatternType>>
{
	static constexpr size_t advance_by = 0;
	static bool matches_impl(Parser &p) { return !PatternType::matches(p); }
};


template<typename PatternType> struct LookAhead : Pattern<LookAhead<PatternType>>
{
	static constexpr size_t advance_by = 0;
	static bool matches_impl(Parser &p)
	{
		// FIXME: this should be done automagically
		const size_t start_position = p.token_position();
		const bool is_match = PatternType::matches(p);
		p.token_position() = start_position;
		return is_match;
	}
};


template<typename lhs, typename rhs> struct AndLiteral : Pattern<AndLiteral<lhs, rhs>>
{
	static constexpr size_t advance_by = lhs::advance_by;

	static bool matches_impl(Parser &p)
	{
		if (lhs::matches(p)) {
			const auto token = p.lexer().peek_token(p.token_position());
			std::string_view token_sv{ token->start().pointer_to_program,
				token->end().pointer_to_program };
			return rhs::matches(token_sv);
		}
		return false;
	}
};


struct StarAtomPattern : Pattern<StarAtomPattern>
{
	// star_atom:
	// | NAME
	// | '(' target_with_star_atom ')'
	// | '(' [star_targets_tuple_seq] ')'
	// | '[' [star_targets_list_seq] ']'
	static bool matches_impl(Parser &p)
	{
		// NAME
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>>;
		if (pattern1::match(p)) {
			spdlog::debug("'(' target_with_star_atom ')'");

			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string id{ p.lexer().get(token->start(), token->end()) };
			p.push_to_stack(std::make_shared<Name>(id, Variable::ContextType::STORE));
			return true;
		}
		return false;
	}
};

struct TargetWithStarAtomPattern : Pattern<TargetWithStarAtomPattern>
{
	static bool matches_impl(Parser &p)
	{
		// star_atom
		using pattern3 = PatternMatch<StarAtomPattern>;
		if (pattern3::match(p)) {
			spdlog::debug("star_atom");
			return true;
		}
		return false;
	}
};


struct StarTargetPattern : Pattern<StarTargetPattern>
{
	static bool matches_impl(Parser &p)
	{
		// target_with_star_atom
		using pattern2 = PatternMatch<TargetWithStarAtomPattern>;
		if (pattern2::match(p)) {
			spdlog::debug("target_with_star_atom");
			return true;
		}
		return false;
	}
};

struct StarTargetsPattern : Pattern<StarTargetsPattern>
{
	// star_targets:
	// | star_target !','
	// | star_target (',' star_target )* [',']
	static bool matches_impl(Parser &p)
	{
		// star_target !','
		using pattern1 = PatternMatch<StarTargetPattern,
			NegativeLookAhead<SingleTokenPattern<Token::TokenType::COMMA>>>;
		if (pattern1::match(p)) {
			spdlog::debug("star_target !','");
			return true;
		}
		using pattern2 = PatternMatch<StarTargetPattern,
			OneOrMorePattern<SingleTokenPattern<Token::TokenType::COMMA>, StarTargetPattern>>;
		if (pattern2::match(p)) {
			spdlog::debug("star_target (',' star_target )* [',']");
			return true;
		}

		return false;
	}
};

struct StringPattern : Pattern<StringPattern>
{
	// strings: STRING+
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::STRING>>;
		if (pattern1::match(p)) {
			spdlog::debug("strings: STRING+");
			// auto token = tokens.begin() - 1;
			auto token = p.lexer().peek_token(p.token_position() - 1);
			// FIXME: this assumes that STRING is surrounded by a single/double quotes
			// 		  could this be """?
			std::string value{ token->start().pointer_to_program + 1,
				token->end().pointer_to_program - 1 };
			p.push_to_stack(std::make_shared<Constant>(value));
			return true;
		}
		return false;
	}
};

struct AtomPattern : Pattern<AtomPattern>
{
	// atom:
	// 	| NAME
	// 	| 'True'
	// 	| 'False'
	// 	| 'None'
	// 	| '__peg_parser__'
	// 	| strings
	// 	| NUMBER
	// 	| (tuple | group | genexp)
	// 	| (list | listcomp)
	// 	| (dict | set | dictcomp | setcomp)
	// 	| '...'
	static bool matches_impl(Parser &p)
	{
		// NAME
		spdlog::debug("atom");
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>>;
		if (pattern1::match(p)) {
			spdlog::debug("NAME");

			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			if (name == "True") {
				p.push_to_stack(std::make_shared<Constant>(true));
			} else if (name == "False") {
				p.push_to_stack(std::make_shared<Constant>(false));
			} else if (name == "None") {
				p.push_to_stack(std::make_shared<Constant>(NoneType{}));
			} else {
				p.push_to_stack(std::make_shared<Name>(name, Variable::ContextType::LOAD));
			}
			return true;
		}
		// strings
		using pattern6 = PatternMatch<OneOrMorePattern<StringPattern>>;
		if (pattern6::match(p)) {
			spdlog::debug("strings");
			return true;
		}

		// NUMBER
		using pattern7 = PatternMatch<SingleTokenPattern<Token::TokenType::NUMBER>>;
		if (pattern7::match(p)) {
			spdlog::debug("NUMBER");
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string number{ token->start().pointer_to_program,
				token->end().pointer_to_program };
			auto dot_iter = std::find(number.begin(), number.end(), '.');
			if (dot_iter == number.end()) {
				// it's an int
				int64_t int_value = std::stoll(number);
				p.push_to_stack(std::make_shared<Constant>(int_value));
			} else {
				double float_value = std::stod(number);
				p.push_to_stack(std::make_shared<Constant>(float_value));
			}

			return true;
		}
		return false;
	}
};

struct ExpressionPattern;

struct NamedExpressionPattern : Pattern<NamedExpressionPattern>
{
	// named_expression:
	// 	| NAME ':=' ~ expression
	// 	| expression !':='
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("NamedExpressionPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		using pattern2 = PatternMatch<ExpressionPattern,
			NegativeLookAhead<SingleTokenPattern<Token::TokenType::COLONEQUAL>>>;
		if (pattern2::match(p)) {
			spdlog::debug("expression !':='");
			return true;
		}
		return false;
	}
};


struct ArgsPattern : Pattern<ArgsPattern>
{
	// args:
	// 	| ','.(starred_expression | named_expression !'=')+ [',' kwargs ]
	// 	| kwargs
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("ArgsPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		using pattern1 = PatternMatch<ApplyInBetweenPattern<NamedExpressionPattern,
			SingleTokenPattern<Token::TokenType::COMMA>>>;
		if (pattern1::match(p)) {
			spdlog::debug("','.(starred_expression | named_expression !'=')+ [',' kwargs ]'");
			spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
			return true;
		}
		return false;
	}
};


struct ArgumentsPattern : Pattern<ArgumentsPattern>
{
	// arguments:
	//     | args [','] &')'
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("ArgumentsPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		using pattern1 = PatternMatch<ArgsPattern,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>>,
			LookAhead<SingleTokenPattern<Token::TokenType::RPAREN>>>;
		if (pattern1::match(p)) {
			spdlog::debug("args [','] &')'");
			return true;
		}
		return false;
	}
};

struct PrimaryPattern_ : Pattern<PrimaryPattern_>
{
	// primary' -> '.' NAME primary'
	// 		| genexp primary'
	// 		| '(' [arguments] ')' primary'
	// 		| '[' slices ']' primary'
	// 		| ϵ
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("PrimaryPattern_");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		const auto original_stack_size = p.stack().size();
		// '(' [arguments] ')' primary'
		using pattern3 = PatternMatch<SingleTokenPattern<Token::TokenType::LPAREN>,
			ZeroOrOnePattern<ArgumentsPattern>,
			SingleTokenPattern<Token::TokenType::RPAREN>,
			PrimaryPattern_>;
		if (pattern3::match(p)) {
			spdlog::debug(" '(' [arguments] ')' primary'");
			const size_t stack_size = p.stack().size();
			std::vector<std::shared_ptr<ASTNode>> args;
			std::vector<std::shared_ptr<ASTNode>> kwargs;
			// FIXME: this takes values from the stack and then pops it later
			// since here we traverse from left to right
			for (size_t i = original_stack_size; i < stack_size; ++i) {
				args.push_back(p.stack()[i]);
			}
			for (size_t i = original_stack_size; i < stack_size; ++i) { p.pop_stack(); }
			auto function = p.pop_stack();
			p.push_to_stack(std::make_shared<Call>(function, args, kwargs));
			p.print_stack();
			return true;
		}

		// ϵ
		using pattern5 = PatternMatch<LookAhead<SingleTokenPattern<Token::TokenType::NEWLINE,
			Token::TokenType::DOUBLESTAR,
			Token::TokenType::STAR,
			Token::TokenType::PLUS,
			Token::TokenType::MINUS,
			Token::TokenType::LEFTSHIFT,
			Token::TokenType::RIGHTSHIFT,
			Token::TokenType::COMMA,
			Token::TokenType::RPAREN>>>;
		if (pattern5::match(p)) { return true; }
		return false;
	}
};

struct PrimaryPattern : Pattern<PrimaryPattern>
{
	// primary:
	//     | invalid_primary  # must be before 'primary genexp' because of invalid_genexp
	//     | primary '.' NAME
	//     | primary genexp
	//     | primary '(' [arguments] ')'
	//     | primary '[' slices ']'
	//     | atom

	// primary -> invalid_primary # must be before 'primary genexp' because of invalid_genexp
	// primary' 		| atom primary'
	static bool matches_impl(Parser &p)
	{
		//  primary' 		| atom primary'
		using pattern2 = PatternMatch<AtomPattern, PrimaryPattern_>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};


struct AwaitPrimaryPattern : Pattern<AwaitPrimaryPattern>
{
	// await_primary:
	//     | AWAIT primary
	//     | primary
	static bool matches_impl(Parser &p)
	{
		// primary
		using pattern2 = PatternMatch<PrimaryPattern>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};

struct FactorPattern;

struct PowerPattern : Pattern<PowerPattern>
{
	// power:
	//     | await_primary '**' factor
	//     | await_primary
	static bool matches_impl(Parser &p)
	{
		// await_primary '**' factor
		using pattern1 = PatternMatch<AwaitPrimaryPattern,
			SingleTokenPattern<Token::TokenType::DOUBLESTAR>,
			FactorPattern>;
		if (pattern1::match(p)) {
			auto rhs = p.pop_stack();
			auto lhs = p.pop_stack();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryExpr::OpType::EXP, lhs, rhs);
			p.push_to_stack(binary_op);
			return true;
		}

		// await_primary
		using pattern2 = PatternMatch<AwaitPrimaryPattern>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};


struct FactorPattern : Pattern<FactorPattern>
{
	// factor:
	//     | '+' factor
	//     | '-' factor
	//     | '~' factor
	//     | power
	static bool matches_impl(Parser &p)
	{
		// power
		using pattern4 = PatternMatch<PowerPattern>;
		if (pattern4::match(p)) { return true; }

		return false;
	}
};

inline std::shared_ptr<ASTNode> &leftmost(std::shared_ptr<ASTNode> &node)
{
	if (auto binop = as<BinaryExpr>(node)) { return leftmost(binop->lhs()); }
	return node;
}


struct TermPattern_ : Pattern<TermPattern_>
{
	// term':
	//     | '*' factor term'
	//     | '/' factor term'
	//     | '//' factor term'
	//     | '%' factor term'
	//     | '@' factor term'
	//     | ϵ
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<SingleTokenPattern<Token::TokenType::STAR>, FactorPattern, TermPattern_>;
		if (pattern1::match(p)) {
			auto rhs = p.pop_stack();
			auto lhs = p.pop_stack();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryExpr::OpType::MULTIPLY, lhs, rhs);
			p.push_to_stack(binary_op);
			return true;
		}
		using pattern2 =
			PatternMatch<SingleTokenPattern<Token::TokenType::SLASH>, FactorPattern, TermPattern_>;
		if (pattern2::match(p)) { return true; }
		// using pattern3 =
		// 	PatternMatch<SingleTokenPattern<Token::TokenType::STAR>, FactorPattern, TermPattern_>;
		// if (pattern3::match(tokens, p)) { return true; }
		// using pattern4 =
		// 	PatternMatch<SingleTokenPattern<Token::TokenType::STAR>, FactorPattern, TermPattern_>;
		// if (pattern4::match(tokens, p)) { return true; }
		// using pattern5 =
		// 	PatternMatch<SingleTokenPattern<Token::TokenType::STAR>, FactorPattern, TermPattern_>;
		// if (pattern5::match(tokens, p)) { return true; }
		// ϵ
		return true;
	}
};


struct TermPattern : Pattern<TermPattern>
{
	// term:
	//     | term '*' factor
	//     | term '/' factor
	//     | term '//' factor
	//     | term '%' factor
	//     | term '@' factor
	//     | factor

	// term: factor term'
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<FactorPattern, TermPattern_>;
		if (pattern1::match(p)) { return true; }

		return false;
	}
};

struct SumPattern_ : Pattern<SumPattern_>
{
	// sum':
	//     | '+' term sum'
	//     | '-' term sum'
	//     | ϵ
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("SumPattern_");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		// '+' term sum'
		using pattern1 =
			PatternMatch<SingleTokenPattern<Token::TokenType::PLUS>, TermPattern, SumPattern_>;
		if (pattern1::match(p)) {
			spdlog::debug("'+' term sum'");
			p.print_stack();
			auto rhs = p.pop_stack();
			auto lhs = p.pop_stack();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryExpr::OpType::PLUS, lhs, rhs);
			p.push_to_stack(binary_op);
			p.print_stack();
			return true;
		}

		// '-' term sum'
		using pattern2 =
			PatternMatch<SingleTokenPattern<Token::TokenType::MINUS>, TermPattern, SumPattern_>;
		if (pattern2::match(p)) {
			spdlog::debug("'-' term sum'");
			auto rhs = p.pop_stack();
			auto lhs = p.pop_stack();
			if (auto rhs_binop = as<BinaryExpr>(rhs)) {
				auto node = std::static_pointer_cast<ASTNode>(rhs_binop);
				auto binary_op =
					std::make_shared<BinaryExpr>(BinaryExpr::OpType::MINUS, lhs, leftmost(node));
				leftmost(node) = binary_op;
				p.push_to_stack(node);
			} else {
				auto binary_op = std::make_shared<BinaryExpr>(BinaryExpr::OpType::MINUS, lhs, rhs);
				p.push_to_stack(binary_op);
			}
			return true;
		}

		// ϵ
		using pattern3 = PatternMatch<LookAhead<SingleTokenPattern<Token::TokenType::NEWLINE,
			Token::TokenType::LEFTSHIFT,
			Token::TokenType::RIGHTSHIFT,
			Token::TokenType::COMMA,
			Token::TokenType::RPAREN>>>;
		if (pattern3::match(p)) { return true; }
		return false;
	}
};


struct SumPattern : Pattern<SumPattern>
{
	// left recursive
	// sum:
	//     | sum '+' term
	//     | sum '-' term
	//     | term

	// sum: term sum'
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("SumPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		// term sum'
		using pattern1 = PatternMatch<TermPattern, SumPattern_>;
		if (pattern1::match(p)) { return true; }

		return false;
	}
};


struct ShiftExprPattern_ : Pattern<ShiftExprPattern_>
{
	// shift_expr': '<<' sum shift_expr'
	//  		  | '>>' sum shift_expr'
	//  		  | ϵ
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("ShiftExprPattern_");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		// '<<' sum shift_expr'
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::LEFTSHIFT>,
			SumPattern,
			ShiftExprPattern_>;
		if (pattern1::match(p)) {
			auto rhs = p.pop_stack();
			auto lhs = p.pop_stack();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryExpr::OpType::LEFTSHIFT, lhs, rhs);
			p.push_to_stack(binary_op);
			return true;
		}

		// '>>' sum shift_expr'
		using pattern2 = PatternMatch<SingleTokenPattern<Token::TokenType::RIGHTSHIFT>,
			SumPattern,
			ShiftExprPattern_>;
		if (pattern2::match(p)) { return true; }

		// ϵ
		using pattern3 = PatternMatch<LookAhead<SingleTokenPattern<Token::TokenType::NEWLINE,
			Token::TokenType::NUMBER,
			Token::TokenType::COMMA,
			Token::TokenType::RPAREN>>>;
		if (pattern3::match(p)) { return true; }
		return false;
	}
};


struct ShiftExprPattern : Pattern<ShiftExprPattern>
{
	// shift_expr:
	//     | shift_expr '<<' sum
	//     | shift_expr '>>' sum
	//     | sum

	// shift_expr: sum shift_expr'
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("ShiftExprPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		using pattern1 = PatternMatch<SumPattern, ShiftExprPattern_>;
		if (pattern1::match(p)) { return true; }
		return false;
	}
};

struct BitwiseAndPattern : Pattern<BitwiseAndPattern>
{
	// bitwise_and:
	//     | bitwise_and '&' shift_expr
	//     | shift_expr
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("BitwiseAndPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		// shift_expr
		using pattern2 = PatternMatch<ShiftExprPattern>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};

struct BitwiseXorPattern : Pattern<BitwiseXorPattern>
{
	// bitwise_xor:
	//     | bitwise_xor '^' bitwise_and
	//     | bitwise_and
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("BitwiseXorPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		// bitwise_and
		using pattern2 = PatternMatch<BitwiseAndPattern>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};


struct BitwiseOrPattern : Pattern<BitwiseOrPattern>
{
	// bitwise_or:
	//     | bitwise_or '|' bitwise_xor
	//     | bitwise_xor
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("BitwiseOrPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		// bitwise_xor
		using pattern2 = PatternMatch<BitwiseXorPattern>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};


struct ComparissonPattern : Pattern<ComparissonPattern>
{
	// comparison:
	//     | bitwise_or compare_op_bitwise_or_pair+
	//     | bitwise_or
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("ComparissonPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		// bitwise_or
		using pattern2 = PatternMatch<BitwiseOrPattern>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};

struct InversionPattern : Pattern<InversionPattern>
{
	// inversion:
	//     | 'not' inversion
	//     | comparison
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("InversionPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		// comparison
		using pattern2 = PatternMatch<ComparissonPattern>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};

struct ConjunctionPattern : Pattern<ConjunctionPattern>
{
	// conjunction:
	//     | inversion ('and' inversion )+
	//     | inversion
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("ConjunctionPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		// inversion
		using pattern2 = PatternMatch<InversionPattern>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};

struct DisjunctionPattern : Pattern<DisjunctionPattern>
{
	// disjunction:
	//     | conjunction ('or' conjunction )+
	//     | conjunction
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("DisjunctionPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		// conjunction
		using pattern2 = PatternMatch<ConjunctionPattern>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};

struct ExpressionPattern : Pattern<ExpressionPattern>
{
	// expression:
	//     | disjunction 'if' disjunction 'else' expression
	//     | disjunction
	//     | lambdef
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("ExpressionPattern");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
		// disjunction
		using pattern2 = PatternMatch<DisjunctionPattern>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};


struct StarExpressionPattern : Pattern<StarExpressionPattern>
{
	// star_expression:
	//     | '*' bitwise_or
	//     | expression
	static bool matches_impl(Parser &p)
	{
		// expression
		using pattern2 = PatternMatch<ExpressionPattern>;
		if (pattern2::match(p)) { return true; }
		return false;
	}
};

struct StarExpressionsPattern : Pattern<StarExpressionsPattern>
{
	// star_expressions:
	// 	| star_expression (',' star_expression )+ [',']
	// 	| star_expression ','
	// 	| star_expression
	static bool matches_impl(Parser &p)
	{
		// star_expression ','
		// using pattern2 =
		// 	PatternMatch<StarExpressionPattern, SingleTokenPattern<Token::TokenType::COMMA>>;
		// if (pattern2::match(tokens, p)) { return true; }
		// star_expression
		using pattern3 = PatternMatch<StarExpressionPattern>;
		if (pattern3::match(p)) { return true; }
		return false;
	}
};

struct AssignmentPattern : Pattern<AssignmentPattern>
{
	// assignment:
	// 	| NAME ':' expression ['=' annotated_rhs ]
	// 	| ('(' single_target ')'
	// 		| single_subscript_attribute_target) ':' expression ['=' annotated_rhs ]
	// 	| (star_targets '=' )+ (yield_expr | star_expressions) !'=' [TYPE_COMMENT]
	// 	| single_target augassign ~ (yield_expr | star_expressions)
	static bool matches_impl(Parser &p)
	{
		using EqualMatch = SingleTokenPattern<Token::TokenType::EQUAL>;
		using pattern3 =
			PatternMatch<OneOrMorePattern<StarTargetsPattern, EqualMatch>, StarExpressionsPattern>;
		size_t start_position = p.stack().size();
		if (pattern3::match(p)) {
			spdlog::debug(
				"(star_targets '=' )+ (yield_expr | star_expressions) !'=' [TYPE_COMMENT]");
			std::vector<std::shared_ptr<Variable>> targets;
			auto expressions = p.pop_stack();
			const auto &stack = p.stack();
			for (size_t i = start_position; i < stack.size(); ++i) {
				targets.push_back(std::static_pointer_cast<Variable>(stack[i]));
			}
			while (p.stack().size() > start_position) { p.pop_stack(); }
			p.print_stack();
			auto assignment = std::make_shared<Assign>(targets, expressions, "");
			p.push_to_stack(assignment);
			return true;
		}
		return false;
	}
};

struct ReturnPattern
{
	static bool matches(std::string_view token_value) { return token_value == "return"; }
};


struct ReturnStatementPattern : Pattern<ReturnStatementPattern>
{
	// return_stmt:
	// 		| 'return' [star_expressions]
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ReturnPattern>,
				ZeroOrMorePattern<StarExpressionsPattern>>;
		if (pattern1::match(p)) {
			const auto &return_value = p.pop_stack();
			auto return_node = std::make_shared<Return>(return_value);
			p.push_to_stack(return_node);
			return true;
		}
		return false;
	}
};

struct SmallStatementPattern : Pattern<SmallStatementPattern>
{
	// small_stmt:
	// 	| assignment
	// 	| star_expressions
	// 	| return_stmt
	// 	| import_stmt
	// 	| raise_stmt
	// 	| 'pass'
	// 	| del_stmt
	// 	| yield_stmt
	// 	| assert_stmt
	// 	| 'break'
	// 	| 'continue'
	// 	| global_stmt
	// 	| nonlocal_stmt
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<AssignmentPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("assignment");
			return true;
		}
		using pattern2 = PatternMatch<StarExpressionsPattern>;
		if (pattern2::match(p)) {
			spdlog::debug("star_expressions");
			return true;
		}
		using pattern3 = PatternMatch<ReturnStatementPattern>;
		if (pattern3::match(p)) {
			spdlog::debug("return_stmt");
			return true;
		}
		return false;
	}
};


struct SimpleStatementPattern : Pattern<SimpleStatementPattern>
{
	// simple_stmt:
	// 	| small_stmt !';' NEWLINE  # Not needed, there for speedup
	// 	| ';'.small_stmt+ [';'] NEWLINE
	static bool matches_impl(Parser &p)
	{
		using EndPattern = SingleTokenPattern<Token::TokenType::NEWLINE, Token::TokenType::SEMI>;

		using pattern1 = PatternMatch<SmallStatementPattern, EndPattern>;

		if (pattern1::match(p)) { return true; }

		return false;
	}
};

struct DefPattern
{
	static bool matches(std::string_view token_value) { return token_value == "def"; }
};


struct ParamPattern : Pattern<ParamPattern>
{
	// param: NAME annotation?
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>>;
		if (pattern1::match(p)) {
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string argname{ token->start().pointer_to_program,
				token->end().pointer_to_program };
			std::string annotation = "";
			p.push_to_stack(std::make_shared<Argument>(argname, annotation, ""));
			return true;
		}
		return false;
	}
};


struct ParamNoDefaultPattern : Pattern<ParamNoDefaultPattern>
{
	// param_no_default:
	//     | param ',' TYPE_COMMENT?
	//     | param TYPE_COMMENT? &')'
	static bool matches_impl(Parser &p)
	{
		// param ',' TYPE_COMMENT?
		// TODO: implement TYPE_COMMENT?
		using pattern1 = PatternMatch<ParamPattern, SingleTokenPattern<Token::TokenType::COMMA>>;
		if (pattern1::match(p)) {
			spdlog::debug("param ',' TYPE_COMMENT?");
			auto arg = p.pop_stack();
			auto args = p.stack().back();
			ASSERT(as<Arguments>(args));
			ASSERT(as<Argument>(arg));
			as<Arguments>(args)->push_arg(as<Argument>(arg));
			return true;
		}

		// param TYPE_COMMENT? &')'
		// TODO: implement TYPE_COMMENT?
		using pattern2 =
			PatternMatch<ParamPattern, LookAhead<SingleTokenPattern<Token::TokenType::RPAREN>>>;
		if (pattern2::match(p)) {
			spdlog::debug("param TYPE_COMMENT? &')'");
			auto arg = p.pop_stack();
			auto args = p.stack().back();
			ASSERT(as<Arguments>(args));
			ASSERT(as<Argument>(arg));
			as<Arguments>(args)->push_arg(as<Argument>(arg));
			return true;
		}
		return false;
	}
};


struct ParametersPattern : Pattern<ParametersPattern>
{
	// parameters:
	//     | slash_no_default param_no_default* param_with_default* [star_etc]
	//     | slash_with_default param_with_default* [star_etc]
	//     | param_no_default+ param_with_default* [star_etc]
	//     | param_with_default+ [star_etc]
	//     | star_etc
	static bool matches_impl(Parser &p)
	{
		p.push_to_stack(std::make_shared<Arguments>());
		using pattern3 = PatternMatch<OneOrMorePattern<ParamNoDefaultPattern>>;
		if (pattern3::match(p)) {
			spdlog::debug("param_no_default+ param_with_default* [star_etc]");
			// p.push_to_stack();
			p.print_stack();
			return true;
		}
		return false;
	}
};


struct ParamsPattern : Pattern<ParamsPattern>
{
	// params:
	// 		| parameters
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<ParametersPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("parameters");
			p.print_stack();
			return true;
		}
		return false;
	}
};

struct StatementsPattern;

struct BlockPattern : Pattern<BlockPattern>
{
	static bool matches_impl(Parser &p)
	{
		// 	block:
		//	 	| NEWLINE INDENT statements DEDENT
		// 		| simple_stmt
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NEWLINE>,
			SingleTokenPattern<Token::TokenType::INDENT>,
			StatementsPattern,
			SingleTokenPattern<Token::TokenType::DEDENT>>;
		if (pattern1::match(p)) {
			spdlog::debug("NEWLINE INDENT statements DEDENT");
			for (const auto &node : p.statements()) { node->print_node(""); }
			return true;
		}

		using pattern2 = PatternMatch<SimpleStatementPattern>;
		if (pattern2::match(p)) {
			spdlog::debug("simple_stmt");
			return true;
		}
		return false;
	}
};


struct FunctionNamePattern : Pattern<FunctionNamePattern>
{
	// function_name: NAME
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>>;
		if (pattern1::match(p)) {
			spdlog::debug("function_name: NAME");
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string function_name{ token->start().pointer_to_program,
				token->end().pointer_to_program };
			p.push_to_stack(std::make_shared<Constant>(function_name));
			return true;
		}
		return false;
	}
};

struct FunctionDefinitionPattern : Pattern<FunctionDefinitionPattern>
{
	// function_def: 'def' function_name '(' [params] ')' ['->' expression ] ':' [func_type_comment]
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, DefPattern>,
				FunctionNamePattern,
				SingleTokenPattern<Token::TokenType::LPAREN>,
				ZeroOrMorePattern<ParamsPattern>,
				SingleTokenPattern<Token::TokenType::RPAREN>,
				SingleTokenPattern<Token::TokenType::COLON>>;
		if (pattern1::match(p)) {
			spdlog::debug(
				"function_def: 'def' function_name '(' [params] ')' ['->' expression ] ':' "
				"[func_type_comment]");
			return true;
		}
		return false;
	}
};


struct FunctionDefinitionRawStatement : Pattern<FunctionDefinitionRawStatement>
{
	// function_def_raw:
	//     | function_def block
	//     | ASYNC 'def' function_name '(' [params] ')' ['->' expression ] ':' [func_type_comment]
	//     block
	static bool matches_impl(Parser &p)
	{
		// function_def_raw
		using pattern1 = PatternMatch<FunctionDefinitionPattern, BlockPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("function_def_raw: function_def block");
			auto args = p.pop_stack();
			auto name = p.pop_stack();
			args->print_node("");
			name->print_node("");

			ASSERT(as<Constant>(name));
			ASSERT(as<Arguments>(args));
			auto function = std::make_shared<FunctionDefinition>(
				std::get<String>(as<Constant>(name)->value()).s,
				as<Arguments>(args),
				std::move(p.statements()),
				std::vector<std::shared_ptr<ASTNode>>{},
				nullptr,
				"");
			p.push_to_stack(function);
			return true;
		}
		return false;
	}
};


struct FunctionDefinitionStatement : Pattern<FunctionDefinitionStatement>
{
	// function_def:
	//     | decorators function_def_raw
	//     | function_def_raw
	static bool matches_impl(Parser &p)
	{
		// function_def_raw
		using pattern2 = PatternMatch<FunctionDefinitionRawStatement>;
		if (pattern2::match(p)) {
			spdlog::debug("function_def_raw");
			return true;
		}
		return false;
	}
};


struct CompoundStatementPattern : Pattern<CompoundStatementPattern>
{
	// compound_stmt:
	//     | function_def
	//     | if_stmt
	//     | class_def
	//     | with_stmt
	//     | for_stmt
	//     | try_stmt
	//     | while_stmt
	static bool matches_impl(Parser &p)
	{
		// function_def
		using pattern1 = PatternMatch<FunctionDefinitionStatement>;
		if (pattern1::match(p)) {
			spdlog::debug("function_def");
			p.print_stack();
			return true;
		}
		return false;
	}
};


struct StatementPattern : Pattern<StatementPattern>
{
	// statement: compound_stmt
	// 	| simple_stmt
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<CompoundStatementPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("compound_stmt");
			return true;
		}
		using pattern2 = PatternMatch<SimpleStatementPattern>;
		if (pattern2::match(p)) {
			spdlog::debug("simple_stmt");
			return true;
		}
		return false;
	}
};


struct StatementsPattern : Pattern<StatementsPattern>
{
	// statements: statement+
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<StatementPattern>;
		while (pattern1::match(p)) {
			p.statements().push_back(p.pop_stack());
			p.statements().back()->print_node("");
			p.commit();
		}
		return true;
	}
};

struct FilePattern : Pattern<FilePattern>
{
	// file: [statements] ENDMARKER
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<StatementsPattern, SingleTokenPattern<Token::TokenType::ENDMARKER>>;
		if (pattern1::match(p)) {
			for (const auto &node : p.statements()) { p.module()->emplace(node); }
			return true;
		}
		PARSER_ERROR();
	}
};

namespace parser {
void Parser::parse()
{
	spdlog::debug("Parser return code: {}", FilePattern::matches(*this));
	m_module->print_node("");
}
}// namespace parser