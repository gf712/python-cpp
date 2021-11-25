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
			while (p.stack().size() > start_stack_size) { p.pop_back(); }
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
			while (p.stack().size() > start_stack_size) { p.pop_back(); }
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

		auto original_token_position = p.token_position();
		while (InBetweenPattern_::match(p)) {
			if (!MainPatternType_::match(p)) {
				p.token_position() = original_token_position;
				break;
			}
			original_token_position = p.token_position();
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

template<size_t TypeIdx, typename PatternTuple> struct OrPattern_
{
	static bool match(Parser &p)
	{
		if constexpr (TypeIdx == std::tuple_size_v<PatternTuple> - 1) {
			return std::tuple_element_t<TypeIdx, PatternTuple>::matches(p);
		} else {
			if (std::tuple_element_t<TypeIdx, PatternTuple>::matches(p)) {
				return true;
			} else {
				return OrPattern_<TypeIdx + 1, PatternTuple>::match(p);
			}
		}
	}
};


template<typename... PatternTypes> struct OrPattern : Pattern<OrPattern<PatternTypes...>>
{
	static bool matches_impl(Parser &p)
	{
		return OrPattern_<0, std::tuple<PatternTypes...>>::match(p);
	}
};

template<size_t TypeIdx, typename PatternTuple> struct AndPattern_
{
	static bool match(Parser &p)
	{
		auto original_token_position = p.token_position();
		if constexpr (TypeIdx == std::tuple_size_v<PatternTuple> - 1) {
			return std::tuple_element_t<TypeIdx, PatternTuple>::matches(p);
		} else {
			if (!std::tuple_element_t<TypeIdx, PatternTuple>::matches(p)) {
				return false;
			} else {
				// reset to the original token as there was a match
				// and we need to now check if that same token matches
				// the other patterns
				p.token_position() = original_token_position;
				return AndPattern_<TypeIdx + 1, PatternTuple>::match(p);
			}
		}
	}
};

template<typename... PatternTypes> struct AndPattern : Pattern<AndPattern<PatternTypes...>>
{
	static constexpr size_t advance_by = 1;

	static bool matches_impl(Parser &p)
	{
		// TODO: static assert that all patterns advance by the same number of tokens
		return AndPattern_<0, std::tuple<PatternTypes...>>::match(p);
	}
};

template<size_t TypeIdx, typename PatternTuple> struct GroupPatterns_
{
	static bool match(Parser &p)
	{
		if constexpr (TypeIdx == std::tuple_size_v<PatternTuple> - 1) {
			return std::tuple_element_t<TypeIdx, PatternTuple>::matches(p);
		} else {
			if (!std::tuple_element_t<TypeIdx, PatternTuple>::matches(p)) {
				return false;
			} else {
				return GroupPatterns_<TypeIdx + 1, PatternTuple>::match(p);
			}
		}
	}
};


template<typename... PatternTypes> struct GroupPatterns : Pattern<GroupPatterns<PatternTypes...>>
{
	static bool matches_impl(Parser &p)
	{
		return GroupPatterns_<0, std::tuple<PatternTypes...>>::match(p);
	}
};


template<typename PatternsType> struct SingleTokenPattern_
{
	static bool match(Parser &p)
	{
		const auto this_token = p.lexer().peek_token(p.token_position())->token_type();
		if (PatternsType::head == this_token) { return true; }
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
			const size_t size =
				std::distance(token->start().pointer_to_program, token->end().pointer_to_program);
			std::string_view token_sv{ token->start().pointer_to_program, size };
			return rhs::matches(token_sv);
		}
		return false;
	}
};


template<typename lhs, typename rhs> struct AndNotLiteral : Pattern<AndNotLiteral<lhs, rhs>>
{
	static constexpr size_t advance_by = lhs::advance_by;

	static bool matches_impl(Parser &p)
	{
		if (lhs::matches(p)) {
			const auto token = p.lexer().peek_token(p.token_position());
			const size_t size =
				std::distance(token->start().pointer_to_program, token->end().pointer_to_program);
			std::string_view token_sv{ token->start().pointer_to_program, size };
			return !rhs::matches(token_sv);
		}
		return true;
	}
};

struct AsPattern
{
	static bool matches(std::string_view token_value) { return token_value == "as"; }
};

struct AssertPattern
{
	static bool matches(std::string_view token_value) { return token_value == "assert"; }
};


struct ExceptPattern
{
	static bool matches(std::string_view token_value) { return token_value == "except"; }
};

struct FinallyPattern
{
	static bool matches(std::string_view token_value) { return token_value == "finally"; }
};

struct FromPattern
{
	static bool matches(std::string_view token_value) { return token_value == "from"; }
};

struct ImportPattern
{
	static bool matches(std::string_view token_value) { return token_value == "import"; }
};

struct RaisePattern
{
	static bool matches(std::string_view token_value) { return token_value == "raise"; }
};

struct TryPattern
{
	static bool matches(std::string_view token_value) { return token_value == "try"; }
};

struct WhilePattern
{
	static bool matches(std::string_view token_value) { return token_value == "while"; }
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
			p.push_to_stack(std::make_shared<Name>(id, ContextType::STORE));
			return true;
		}
		return false;
	}
};

struct TLookahead : Pattern<TLookahead>
{
	// t_lookahead: '(' | '[' | '.'
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("t_lookahead");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());

		using pattern1 = PatternMatch<OrPattern<SingleTokenPattern<Token::TokenType::LPAREN>,
			SingleTokenPattern<Token::TokenType::LSQB>,
			SingleTokenPattern<Token::TokenType::DOT>>>;
		if (pattern1::match(p)) {
			spdlog::debug("t_lookahead: '(' | '[' | '.'");
			return true;
		}
		return false;
	}
};

struct AtomPattern;

struct TPrimaryPattern : Pattern<TPrimaryPattern>
{
	// t_primary:
	// 	| t_primary '.' NAME &t_lookahead
	// 	| t_primary '[' slices ']' &t_lookahead
	// 	| t_primary genexp &t_lookahead
	// 	| t_primary '(' [arguments] ')' &t_lookahead
	// 	| atom &t_lookahead
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("t_primary");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());

		using pattern5 = PatternMatch<AtomPattern, LookAhead<TLookahead>>;
		if (pattern5::match(p)) {
			spdlog::debug("atom &t_lookahead");
			return true;
		}
		return false;
	}
};

struct SlicesPattern;

struct TargetWithStarAtomPattern : Pattern<TargetWithStarAtomPattern>
{
	// target_with_star_atom:
	// 		| t_primary '.' NAME !t_lookahead
	// 		| t_primary '[' slices ']' !t_lookahead
	// 		| star_atom
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("target_with_star_atom");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());

		// t_primary '.' NAME !t_lookahead
		using pattern1 = PatternMatch<TPrimaryPattern,
			SingleTokenPattern<Token::TokenType::DOT>,
			SingleTokenPattern<Token::TokenType::NAME>,
			NegativeLookAhead<TLookahead>>;
		if (pattern1::match(p)) {
			spdlog::debug("t_primary '.' NAME !t_lookahead");
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			const auto primary = p.pop_back();
			spdlog::debug("{}", name);
			primary->print_node("");
			auto attribute = std::make_shared<Attribute>(primary, name, ContextType::STORE);
			p.push_to_stack(attribute);
			return true;
		}

		using pattern2 = PatternMatch<TPrimaryPattern,
			SingleTokenPattern<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPattern<Token::TokenType::RSQB>,
			NegativeLookAhead<TLookahead>>;
		if (pattern2::match(p)) {
			spdlog::debug("t_primary '[' slices ']' !t_lookahead");
			auto subscript = p.pop_back();
			auto name = p.pop_back();
			ASSERT(as<Subscript>(subscript))
			ASSERT(as<Name>(name))
			as<Name>(name)->set_context(ContextType::LOAD);
			as<Subscript>(subscript)->set_value(name);
			as<Subscript>(subscript)->set_context(ContextType::STORE);
			p.push_to_stack(subscript);
			return true;
		}

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
	// star_target:
	//     | '*' (!'*' star_target)
	//     | target_with_star_atom
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("star_target");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());

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
		spdlog::debug("star_targets");
		spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());

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

struct BitwiseOrPattern;
struct NamedExpressionPattern;

struct StarNamedExpression : Pattern<StarNamedExpression>
{
	// star_named_expression:
	// | '*' bitwise_or
	// | named_expression
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::STAR>, BitwiseOrPattern>;
		if (pattern1::match(p)) { return true; }

		using pattern2 = PatternMatch<NamedExpressionPattern>;
		if (pattern2::match(p)) { return true; }

		return false;
	}
};

struct StarNamedExpressions : Pattern<StarNamedExpressions>
{
	// star_named_expressions: ','.star_named_expression+ [',']
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("star_named_expressions");
		using pattern1 = PatternMatch<
			ApplyInBetweenPattern<StarNamedExpression, SingleTokenPattern<Token::TokenType::COMMA>>,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>>>;

		if (pattern1::match(p)) { return true; }

		return false;
	}
};

struct ListPattern : Pattern<ListPattern>
{
	// list: '[' [star_named_expressions] ']'
	static bool matches_impl(Parser &p)
	{
		BlockScope list_scope{ p };
		spdlog::debug("list");
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::LSQB>,
			ZeroOrMorePattern<StarNamedExpressions>,
			SingleTokenPattern<Token::TokenType::RSQB>>;
		if (pattern1::match(p)) {
			auto list = std::make_shared<List>(ContextType::LOAD);
			while (!p.stack().empty()) { list->append(p.pop_front()); }
			list_scope.parent().push_back(std::move(list));
			return true;
		}
		return false;
	}
};


struct ListCompPattern : Pattern<ListCompPattern>
{
	static bool matches_impl(Parser &) { return false; }
};


struct TuplePattern : Pattern<TuplePattern>
{
	// tuple:
	//     | '(' [star_named_expression ',' [star_named_expressions]  ] ')'
	static bool matches_impl(Parser &p)
	{
		BlockScope list_scope{ p };
		spdlog::debug("list");
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::LPAREN>,
			ZeroOrMorePattern<StarNamedExpression,
				SingleTokenPattern<Token::TokenType::COMMA>,
				ZeroOrMorePattern<StarNamedExpressions>>,
			SingleTokenPattern<Token::TokenType::RPAREN>>;
		if (pattern1::match(p)) {
			auto tuple = std::make_shared<Tuple>(ContextType::LOAD);
			while (!p.stack().empty()) { tuple->append(p.pop_front()); }
			list_scope.parent().push_back(std::move(tuple));
			return true;
		}
		return false;
	}
};


struct GroupPattern : Pattern<GroupPattern>
{
	static bool matches_impl(Parser &) { return false; }
};


struct GenexPattern : Pattern<GenexPattern>
{
	static bool matches_impl(Parser &) { return false; }
};

struct ExpressionPattern;

struct KVPairPattern : Pattern<KVPairPattern>
{
	// kvpair: expression ':' expression
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("kvpair");
		using pattern1 = PatternMatch<ExpressionPattern,
			SingleTokenPattern<Token::TokenType::COLON>,
			ExpressionPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("kvpair: expression ':' expression");
			auto value = p.pop_back();
			auto key = p.pop_back();
			auto dict = p.stack().back();
			ASSERT(as<Dict>(dict))
			as<Dict>(dict)->insert(key, value);
			return true;
		}
		return false;
	}
};


struct DoubleStarredKVPairPattern : Pattern<DoubleStarredKVPairPattern>
{
	// double_starred_kvpair:
	// 		| '**' bitwise_or
	// 		| kvpair
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("double_starred_kvpair");
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::DOUBLESTAR>>;
		if (pattern1::match(p)) {
			spdlog::debug("'**' bitwise_or");
			return true;
		}

		using pattern2 = PatternMatch<KVPairPattern>;
		if (pattern2::match(p)) {
			spdlog::debug("kvpair");
			return true;
		}

		return false;
	}
};

struct DoubleStarredKVPairsPattern : Pattern<DoubleStarredKVPairsPattern>
{
	// double_starred_kvpairs: ','.double_starred_kvpair+ [',']
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("double_starred_kvpairs");
		using pattern1 = PatternMatch<ApplyInBetweenPattern<DoubleStarredKVPairPattern,
										  SingleTokenPattern<Token::TokenType::COMMA>>,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>>>;
		if (pattern1::match(p)) {
			spdlog::debug("','.double_starred_kvpair+ [',']");
			return true;
		}
		return false;
	}
};

struct DictPattern : Pattern<DictPattern>
{
	// dict:
	// | '{' [double_starred_kvpairs] '}'
	// | '{' invalid_double_starred_kvpairs '}'
	static bool matches_impl(Parser &p)
	{
		BlockScope dict_scope{ p };
		p.push_to_stack(std::make_shared<Dict>());
		// '{' [double_starred_kvpairs] '}'
		spdlog::debug("'{' [double_starred_kvpairs] '}'");
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::LBRACE>,
			ZeroOrOnePattern<DoubleStarredKVPairsPattern>,
			SingleTokenPattern<Token::TokenType::RBRACE>>;
		if (pattern1::match(p)) {
			spdlog::debug("'{' [double_starred_kvpairs] '}'");
			dict_scope.parent().push_back(p.pop_back());
			return true;
		}

		return false;
	}
};


struct SetPattern : Pattern<SetPattern>
{
	static bool matches_impl(Parser &) { return false; }
};


struct DictCompPattern : Pattern<DictCompPattern>
{
	static bool matches_impl(Parser &) { return false; }
};


struct SetCompPattern : Pattern<SetCompPattern>
{
	static bool matches_impl(Parser &) { return false; }
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
		spdlog::debug("atom");
		{
			const auto token = p.lexer().peek_token(p.token_position());
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			spdlog::debug(name);
		}
		// NAME
		// using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>>;
		using pattern1 = PatternMatch<AndPattern<SingleTokenPattern<Token::TokenType::NAME>,
			AndNotLiteral<SingleTokenPattern<Token::TokenType::NAME>, RaisePattern>>>;
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
				p.push_to_stack(std::make_shared<Name>(name, ContextType::LOAD));
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

		// 	| (tuple | group | genexp)
		using pattern8 = PatternMatch<OrPattern<TuplePattern, GroupPattern, GenexPattern>>;
		if (pattern8::match(p)) {
			spdlog::debug("(tuple | group | genexp)");
			return true;
		}

		// 	| (list | listcomp)
		using pattern9 = PatternMatch<OrPattern<ListPattern, ListCompPattern>>;
		if (pattern9::match(p)) {
			spdlog::debug("(list | listcomp)");
			return true;
		}

		// (dict | set | dictcomp | setcomp)
		using pattern10 =
			PatternMatch<OrPattern<DictPattern, SetPattern, DictCompPattern, SetCompPattern>>;
		if (pattern10::match(p)) {
			spdlog::debug("(dict | set | dictcomp | setcomp)");
			return true;
		}

		return false;
	}
};


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


struct KwargsOrStarredPattern : Pattern<KwargsOrStarredPattern>
{
	// kwarg_or_starred:
	//     | NAME '=' expression
	//     | starred_expression
	static bool matches_impl(Parser &p)
	{
		const auto token = p.lexer().peek_token(p.token_position());
		std::string maybe_name{ p.lexer().get(token->start(), token->end()) };
		spdlog::debug("kwarg_or_starred");
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>,
			SingleTokenPattern<Token::TokenType::EQUAL>,
			ExpressionPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("NAME '=' expression");
			p.push_to_stack(std::make_shared<Keyword>(maybe_name, p.pop_back()));
			return true;
		}
		return false;
	}
};


struct KwargsPattern : Pattern<KwargsPattern>
{
	// kwargs:
	//     | ','.kwarg_or_starred+ ',' ','.kwarg_or_double_starred+
	//     | ','.kwarg_or_starred+
	//     | ','.kwarg_or_double_starred+
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("kwargs");
		using pattern2 = PatternMatch<OneOrMorePattern<ApplyInBetweenPattern<KwargsOrStarredPattern,
			SingleTokenPattern<Token::TokenType::COMMA>>>>;
		if (pattern2::match(p)) {
			spdlog::debug("','.kwarg_or_starred+");
			return true;
		}
		return false;
	}
};

struct StarredExpressionPattern : Pattern<StarredExpressionPattern>
{
	// starred_expression:
	//     | '*' expression
	static bool matches_impl(Parser &)
	{
		spdlog::debug("StarredExpressionPattern");
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
		spdlog::debug(
			"Testing pattern: ','.(starred_expression | named_expression !'=')+ [',' kwargs ]");
		using pattern1 = PatternMatch<
			ApplyInBetweenPattern<
				OrPattern<StarredExpressionPattern,
					GroupPatterns<NamedExpressionPattern,
						NegativeLookAhead<SingleTokenPattern<Token::TokenType::EQUAL>>>>,
				SingleTokenPattern<Token::TokenType::COMMA>>,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>, KwargsPattern>>;
		if (pattern1::match(p)) {
			spdlog::debug("','.(starred_expression | named_expression !'=')+ [',' kwargs ]'");
			spdlog::debug("{}", p.lexer().peek_token(p.token_position())->to_string());
			return true;
		}
		using pattern2 = PatternMatch<KwargsPattern>;
		spdlog::debug("Testing pattern: kwargs");
		if (pattern2::match(p)) {
			spdlog::debug("kwargs");
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

struct SlicePattern : Pattern<SlicePattern>
{
	// slice:
	//     | [expression] ':' [expression] [':' [expression] ]
	//     | named_expression
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("SlicePattern");
		using pattern1 = PatternMatch<ZeroOrOnePattern<ExpressionPattern>,
			SingleTokenPattern<Token::TokenType::COLON>,
			ZeroOrOnePattern<ExpressionPattern>,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COLON>,
				ZeroOrOnePattern<ExpressionPattern>>>;
		const auto initial_size = p.stack().size();
		if (pattern1::match(p)) {
			spdlog::debug("[expression] ':' [expression] [':' [expression] ]");
			Subscript::SliceType slice;
			if ((p.stack().size() - initial_size) > 1) {
				if ((p.stack().size() - initial_size) == 3) {
					auto step = p.pop_back();
					auto upper = p.pop_back();
					auto lower = p.pop_back();
					slice = Subscript::Slice{ lower, upper, step };
				} else if ((p.stack().size() - initial_size) == 2) {
					auto upper = p.pop_back();
					auto lower = p.pop_back();
					slice = Subscript::Slice{ lower, upper };
				} else {
					PARSER_ERROR()
				}
			} else if ((p.stack().size() - initial_size) == 1) {
				slice = Subscript::Index{ p.pop_back() };
			} else {
				PARSER_ERROR()
			}
			as<ast::Subscript>(p.stack().back())->set_slice(slice);
			return true;
		}

		using pattern2 = PatternMatch<NamedExpressionPattern>;
		if (pattern2::match(p)) {
			spdlog::debug("named_expression");
			Subscript::SliceType slice = Subscript::Index{ p.pop_back() };
			as<ast::Subscript>(p.stack().back())->set_slice(slice);
			return true;
		}
		return false;
	}
};


struct SlicesPattern : Pattern<SlicesPattern>
{
	// slices:
	//     | slice !','
	//     | ','.slice+ [',']
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("SlicesPattern");
		p.push_to_stack(std::make_shared<Subscript>());

		using pattern1 = PatternMatch<SlicePattern,
			NegativeLookAhead<SingleTokenPattern<Token::TokenType::COMMA>>>;
		if (pattern1::match(p)) {
			spdlog::debug("slice !','");
			return true;
		}

		using pattern2 = PatternMatch<
			ApplyInBetweenPattern<SlicePattern, SingleTokenPattern<Token::TokenType::DOT>>,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>>>;
		if (pattern2::match(p)) {
			spdlog::debug("','.slice+ [',']");
			return true;
		}

		p.pop_back();
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
		BlockScope primary_scope{ p };
		// '.' NAME primary'
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::DOT>,
			SingleTokenPattern<Token::TokenType::NAME>>;
		if (pattern1::match(p)) {
			spdlog::debug("'.' NAME");
			auto value = primary_scope.parent().back();
			primary_scope.parent().pop_back();
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			using pattern1a = PatternMatch<PrimaryPattern_>;
			p.push_to_stack(std::make_shared<Attribute>(value, name, ContextType::LOAD));
			if (pattern1a::match(p)) {
				spdlog::debug("'.' NAME primary'");
				primary_scope.parent().push_back(p.pop_back());
				return true;
			}
			return false;
		}

		using pattern3 = PatternMatch<SingleTokenPattern<Token::TokenType::LPAREN>,
			ZeroOrOnePattern<ArgumentsPattern>,
			SingleTokenPattern<Token::TokenType::RPAREN>,
			PrimaryPattern_>;
		if (pattern3::match(p)) {
			spdlog::debug("'(' [arguments] ')' primary'");
			p.print_stack();
			spdlog::debug("--------------");
			std::vector<std::shared_ptr<ASTNode>> args;
			std::vector<std::shared_ptr<Keyword>> kwargs;
			for (const auto &node : p.stack()) {
				if (auto keyword_node = as<Keyword>(node)) {
					kwargs.push_back(keyword_node);
				} else {
					args.push_back(node);
				}
			}
			auto function = primary_scope.parent().back();
			primary_scope.parent().pop_back();
			primary_scope.parent().push_back(std::make_shared<Call>(function, args, kwargs));
			p.print_stack();
			return true;
		}

		// '[' slices ']' primary'
		using pattern4 = PatternMatch<SingleTokenPattern<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPattern<Token::TokenType::RSQB>>;
		if (pattern4::match(p)) {
			spdlog::debug("'[' slices ']' primary'");
			ASSERT(as<Subscript>(p.stack().back()))
			primary_scope.parent().push_back(p.pop_back());
			return true;
		}

		// ϵ
		using pattern5 =
			PatternMatch<LookAhead<OrPattern<SingleTokenPattern<Token::TokenType::NEWLINE,
												 Token::TokenType::DOUBLESTAR,
												 Token::TokenType::STAR,
												 Token::TokenType::PLUS,
												 Token::TokenType::MINUS,
												 Token::TokenType::LEFTSHIFT,
												 Token::TokenType::PERCENT,
												 Token::TokenType::RIGHTSHIFT,
												 Token::TokenType::COMMA,
												 Token::TokenType::RPAREN,
												 Token::TokenType::COLON,
												 Token::TokenType::EQUAL,
												 Token::TokenType::EQEQUAL,
												 Token::TokenType::NOTEQUAL,
												 Token::TokenType::LESSEQUAL,
												 Token::TokenType::LESS,
												 Token::TokenType::GREATEREQUAL,
												 Token::TokenType::GREATER,
												 Token::TokenType::RSQB,
												 Token::TokenType::RBRACE>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, FromPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, AsPattern>>>>;
		if (pattern5::match(p)) {
			spdlog::debug("ϵ");
			return true;
		}
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
		if (pattern2::match(p)) {
			if (as<Subscript>(p.stack().back())) {
				auto subscript = p.pop_back();
				auto value = p.pop_back();
				ASSERT(as<Subscript>(subscript))
				ASSERT(as<Name>(value))
				as<Subscript>(subscript)->set_value(as<Name>(value));
				as<Subscript>(subscript)->set_context(ContextType::LOAD);
				p.push_to_stack(subscript);
			}
			return true;
		}

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
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::EXP, lhs, rhs);
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
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::MULTIPLY, lhs, rhs);
			p.push_to_stack(binary_op);
			return true;
		}
		using pattern2 =
			PatternMatch<SingleTokenPattern<Token::TokenType::SLASH>, FactorPattern, TermPattern_>;
		if (pattern2::match(p)) { return true; }
		// using pattern3 =
		// 	PatternMatch<SingleTokenPattern<Token::TokenType::STAR>, FactorPattern, TermPattern_>;
		// if (pattern3::match(tokens, p)) { return true; }
		using pattern4 = PatternMatch<SingleTokenPattern<Token::TokenType::PERCENT>,
			FactorPattern,
			TermPattern_>;
		if (pattern4::match(p)) {
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::MODULO, lhs, rhs);
			p.push_to_stack(binary_op);
			return true;
		}
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
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::PLUS, lhs, rhs);
			p.push_to_stack(binary_op);
			p.print_stack();
			return true;
		}

		// '-' term sum'
		using pattern2 =
			PatternMatch<SingleTokenPattern<Token::TokenType::MINUS>, TermPattern, SumPattern_>;
		if (pattern2::match(p)) {
			spdlog::debug("'-' term sum'");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			if (auto rhs_binop = as<BinaryExpr>(rhs)) {
				auto node = std::static_pointer_cast<ASTNode>(rhs_binop);
				auto binary_op =
					std::make_shared<BinaryExpr>(BinaryOpType::MINUS, lhs, leftmost(node));
				leftmost(node) = binary_op;
				p.push_to_stack(node);
			} else {
				auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::MINUS, lhs, rhs);
				p.push_to_stack(binary_op);
			}
			return true;
		}

		// ϵ
		using pattern3 = PatternMatch<LookAhead<SingleTokenPattern<Token::TokenType::NEWLINE,
			Token::TokenType::LEFTSHIFT,
			Token::TokenType::RIGHTSHIFT,
			Token::TokenType::COMMA,
			Token::TokenType::RPAREN,
			Token::TokenType::NAME,
			Token::TokenType::COLON,
			Token::TokenType::EQEQUAL,
			Token::TokenType::NOTEQUAL,
			Token::TokenType::LESSEQUAL,
			Token::TokenType::LESS,
			Token::TokenType::GREATEREQUAL,
			Token::TokenType::GREATER,
			Token::TokenType::EQUAL,
			Token::TokenType::RSQB,
			Token::TokenType::RBRACE>>>;
		if (pattern3::match(p)) {
			spdlog::debug("ϵ");
			return true;
		}
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
		if (pattern1::match(p)) {
			spdlog::debug("term sum'");
			return true;
		}

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
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::LEFTSHIFT, lhs, rhs);
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
			Token::TokenType::RPAREN,
			Token::TokenType::NAME,
			Token::TokenType::COLON,
			Token::TokenType::EQEQUAL,
			Token::TokenType::NOTEQUAL,
			Token::TokenType::LESSEQUAL,
			Token::TokenType::LESS,
			Token::TokenType::GREATEREQUAL,
			Token::TokenType::GREATER,
			Token::TokenType::EQUAL,
			Token::TokenType::RSQB,
			Token::TokenType::RBRACE>>>;
		if (pattern3::match(p)) {
			spdlog::debug("shift_expr: ϵ");
			return true;
		}
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
		if (pattern1::match(p)) {
			spdlog::debug("shift_expr '<<' sum");
			return true;
		}
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
		if (pattern2::match(p)) {
			spdlog::debug("shift_expr");
			return true;
		}

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
		if (pattern2::match(p)) {
			spdlog::debug("bitwise_and");
			return true;
		}

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
		if (pattern2::match(p)) {
			spdlog::debug("bitwise_xor");
			return true;
		}

		return false;
	}
};

struct EqBitwiseOrPattern : Pattern<EqBitwiseOrPattern>
{
	// eq_bitwise_or: '==' bitwise_or
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<SingleTokenPattern<Token::TokenType::EQEQUAL>, BitwiseOrPattern>;
		spdlog::debug("EqBitwiseOrPattern");
		if (pattern1::match(p)) {
			spdlog::debug("'==' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs, Compare::OpType::Eq, rhs);
			p.push_to_stack(comparisson);
			return true;
		}
		return false;
	}
};


struct NotEqBitwiseOrPattern : Pattern<NotEqBitwiseOrPattern>
{
	// noteq_bitwise_or: '!=' bitwise_or
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<SingleTokenPattern<Token::TokenType::NOTEQUAL>, BitwiseOrPattern>;
		spdlog::debug("NotEqBitwiseOrPattern");
		if (pattern1::match(p)) {
			spdlog::debug("'!=' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs, Compare::OpType::NotEq, rhs);
			p.push_to_stack(comparisson);
			return true;
		}
		return false;
	}
};

struct LtEqBitwiseOrPattern : Pattern<LtEqBitwiseOrPattern>
{
	// lteq_bitwise_or: '<=' bitwise_or
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<SingleTokenPattern<Token::TokenType::LESSEQUAL>, BitwiseOrPattern>;
		spdlog::debug("LtEqBitwiseOrPattern");
		if (pattern1::match(p)) {
			spdlog::debug("'<=' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs, Compare::OpType::LtE, rhs);
			p.push_to_stack(comparisson);
			return true;
		}
		return false;
	}
};

struct LtBitwiseOrPattern : Pattern<LtBitwiseOrPattern>
{
	// lteq_bitwise_or: '<' bitwise_or
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::LESS>, BitwiseOrPattern>;
		spdlog::debug("LtBitwiseOrPattern");
		if (pattern1::match(p)) {
			spdlog::debug("'<' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs, Compare::OpType::Lt, rhs);
			p.push_to_stack(comparisson);
			return true;
		}
		return false;
	}
};

struct GtEqBitwiseOrPattern : Pattern<GtEqBitwiseOrPattern>
{
	// lteq_bitwise_or: '<' bitwise_or
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<SingleTokenPattern<Token::TokenType::GREATEREQUAL>, BitwiseOrPattern>;
		spdlog::debug("GtEqBitwiseOrPattern");
		if (pattern1::match(p)) {
			spdlog::debug("'>=' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs, Compare::OpType::GtE, rhs);
			p.push_to_stack(comparisson);
			return true;
		}
		return false;
	}
};

struct GtBitwiseOrPattern : Pattern<GtBitwiseOrPattern>
{
	// lteq_bitwise_or: '<' bitwise_or
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<SingleTokenPattern<Token::TokenType::GREATER>, BitwiseOrPattern>;
		spdlog::debug("GtBitwiseOrPattern");
		if (pattern1::match(p)) {
			spdlog::debug("'>' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs, Compare::OpType::Gt, rhs);
			p.push_to_stack(comparisson);
			return true;
		}
		return false;
	}
};

struct CompareOpBitwiseOrPairPattern : Pattern<CompareOpBitwiseOrPairPattern>
{
	// compare_op_bitwise_or_pair:
	//     | eq_bitwise_or
	//     | noteq_bitwise_or
	//     | lte_bitwise_or
	//     | lt_bitwise_or
	//     | gte_bitwise_or
	//     | gt_bitwise_or
	//     | notin_bitwise_or
	//     | in_bitwise_or
	//     | isnot_bitwise_or
	//     | is_bitwise_or
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("CompareOpBitwiseOrPairPattern");
		// eq_bitwise_or
		using pattern1 = PatternMatch<EqBitwiseOrPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("eq_bitwise_or");
			return true;
		}
		// noteq_bitwise_or
		using pattern2 = PatternMatch<NotEqBitwiseOrPattern>;
		if (pattern2::match(p)) {
			spdlog::debug("not_eq_bitwise_or");
			return true;
		}
		// lte_bitwise_or
		using pattern3 = PatternMatch<LtEqBitwiseOrPattern>;
		if (pattern3::match(p)) {
			spdlog::debug("lte_bitwise_or");
			return true;
		}
		// lt_bitwise_or
		using pattern4 = PatternMatch<LtBitwiseOrPattern>;
		if (pattern4::match(p)) {
			spdlog::debug("lt_bitwise_or");
			return true;
		}
		// gte_bitwise_or
		using pattern5 = PatternMatch<GtEqBitwiseOrPattern>;
		if (pattern5::match(p)) {
			spdlog::debug("gte_bitwise_or");
			return true;
		}
		// gt_bitwise_or
		using pattern6 = PatternMatch<GtBitwiseOrPattern>;
		if (pattern6::match(p)) {
			spdlog::debug("gt_bitwise_or");
			return true;
		}
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
		// bitwise_or compare_op_bitwise_or_pair+
		using pattern1 =
			PatternMatch<BitwiseOrPattern, OneOrMorePattern<CompareOpBitwiseOrPairPattern>>;
		if (pattern1::match(p)) {
			spdlog::debug("bitwise_or compare_op_bitwise_or_pair+");
			return true;
		}

		// bitwise_or
		using pattern2 = PatternMatch<BitwiseOrPattern>;
		if (pattern2::match(p)) {
			spdlog::debug("bitwise_or");
			return true;
		}

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
		size_t initial_stack_size = p.stack().size();
		// star_expression (',' star_expression )+ [',']
		using pattern1 = PatternMatch<StarExpressionPattern,
			OneOrMorePattern<SingleTokenPattern<Token::TokenType::COMMA>, StarExpressionPattern>,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>>>;
		if (pattern1::match(p)) {
			spdlog::debug("star_expression (',' star_expression )+ [',']");
			auto result = std::make_shared<Tuple>(ContextType::LOAD);
			std::vector<std::shared_ptr<ASTNode>> expressions;
			while (p.stack().size() > initial_stack_size) { expressions.push_back(p.pop_back()); }
			std::for_each(expressions.rbegin(), expressions.rend(), [&result](const auto &el) {
				result->append(el);
			});
			p.push_to_stack(result);
			return true;
		}
		// star_expression ','
		using pattern2 =
			PatternMatch<StarExpressionPattern, SingleTokenPattern<Token::TokenType::COMMA>>;
		if (pattern2::match(p)) {
			spdlog::debug("star_expression ','");
			return true;
		}
		// star_expression
		using pattern3 = PatternMatch<StarExpressionPattern>;
		if (pattern3::match(p)) {
			spdlog::debug("star_expression");
			return true;
		}
		return false;
	}
};


struct SingleTargetPattern : Pattern<SingleTargetPattern>
{
	// single_target:
	//     | single_subscript_attribute_target
	//     | NAME
	//     | '(' single_target ')'
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("SingleTargetPattern");
		using pattern2 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>>;
		if (pattern2::match(p)) {
			spdlog::debug("NAME");
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			p.push_to_stack(std::make_shared<Name>(name, ContextType::STORE));
			return true;
		}

		using pattern3 = PatternMatch<SingleTokenPattern<Token::TokenType::LPAREN>,
			SingleTargetPattern,
			SingleTokenPattern<Token::TokenType::RPAREN>>;
		if (pattern3::match(p)) {
			spdlog::debug("'(' single_target ')'");
			return true;
		}

		return false;
	}
};


struct AugAssignPattern : Pattern<AugAssignPattern>
{
	// augassign:
	//     | '+='
	//     | '-='
	//     | '*='
	//     | '@='
	//     | '/='
	//     | '%='
	//     | '&='
	//     | '|='
	//     | '^='
	//     | '<<='
	//     | '>>='
	//     | '**='
	//     | '//='
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::PLUSEQUAL>>;
		if (pattern1::match(p)) {
			spdlog::debug("'+='");
			const auto &lhs = p.pop_back();
			// defer rhs assignment to caller. Am I shooting myself in the foot?
			// at least a null dereference goes with a bang...
			p.push_to_stack(std::make_shared<AugAssign>(lhs, BinaryOpType::PLUS, nullptr));
			return true;
		}

		return false;
	}
};


struct YieldExpressionPattern : Pattern<YieldExpressionPattern>
{
	static bool matches_impl(Parser &) { return false; }
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
			auto targets = std::make_shared<Tuple>(ContextType::STORE);
			auto expressions = p.pop_back();
			const auto &stack = p.stack();
			for (size_t i = start_position; i < stack.size(); ++i) { targets->append(stack[i]); }
			while (p.stack().size() > start_position) { p.pop_back(); }
			p.print_stack();
			expressions->print_node("");

			auto assignment = [&]() {
				if (targets->elements().size() == 1) {
					return std::make_shared<Assign>(
						std::vector<std::shared_ptr<ASTNode>>{ targets->elements().back() },
						expressions,
						"");
				} else {
					return std::make_shared<Assign>(
						std::vector<std::shared_ptr<ASTNode>>{ targets }, expressions, "");
				}
			}();
			p.push_to_stack(assignment);
			assignment->print_node("");
			return true;
		}

		using pattern4 = PatternMatch<SingleTargetPattern,
			AugAssignPattern,
			OrPattern<YieldExpressionPattern, StarExpressionsPattern>>;
		if (pattern4::match(p)) {
			spdlog::debug("single_target augassign ~ (yield_expr | star_expressions)");
			const auto &rhs = p.pop_back();
			auto aug_assign = p.pop_back();
			as<AugAssign>(aug_assign)->set_value(rhs);
			p.push_to_stack(aug_assign);
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
			const auto &return_value = p.pop_back();
			auto return_node = std::make_shared<Return>(return_value);
			p.push_to_stack(return_node);
			return true;
		}
		return false;
	}
};


struct DottedNamePattern_ : Pattern<DottedNamePattern_>
{
	//  dotted_name':
	//	   | '.' NAME dotted_name'
	//	   | ϵ
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("dotted_name");
		const auto token = p.lexer().peek_token(p.token_position() + 1);
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::DOT>,
			SingleTokenPattern<Token::TokenType::NAME>,
			DottedNamePattern_>;
		if (pattern1::match(p)) {
			spdlog::debug("'.' NAME dotted_name'");
			std::string name{ p.lexer().get(token->start(), token->end()) };
			p.push_to_stack(std::make_shared<Constant>(name));
			return true;
		}
		using pattern2 = PatternMatch<
			LookAhead<SingleTokenPattern<Token::TokenType::NEWLINE, Token::TokenType::NAME>>>;
		if (pattern2::match(p)) {
			spdlog::debug("ϵ");
			return true;
		}

		return false;
	}
};

struct DottedNamePattern : Pattern<DottedNamePattern>
{
	//  dotted_name:
	// 	   NAME dotted_name'
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("dotted_name");
		const auto token = p.lexer().peek_token(p.token_position());
		std::string name{ p.lexer().get(token->start(), token->end()) };
		size_t stack_position = p.stack().size();
		using pattern2 =
			PatternMatch<SingleTokenPattern<Token::TokenType::NAME>, DottedNamePattern_>;
		if (pattern2::match(p)) {
			spdlog::debug("NAME dotted_name'");
			std::static_pointer_cast<Import>(p.stack()[stack_position - 1])->add_dotted_name(name);
			while (p.stack().size() > stack_position) {
				const auto &node = p.pop_back();
				std::string value =
					std::get<String>(static_pointer_cast<Constant>(node)->value()).s;
				std::static_pointer_cast<Import>(p.stack()[stack_position - 1])
					->add_dotted_name(value);
			}
			return true;
		}
		return false;
	}
};

struct DottedAsNamePattern : Pattern<DottedAsNamePattern>
{
	// dotted_as_name:
	//     | dotted_name' ['as' NAME ]
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("dotted_as_name");
		using pattern1 = PatternMatch<DottedNamePattern>;
		if (pattern1::match(p)) {
			using pattern1a =
				PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, AsPattern>,
					SingleTokenPattern<Token::TokenType::NAME>>;
			if (pattern1a::match(p)) {
				spdlog::debug("['as' NAME ]");
				const auto token = p.lexer().peek_token(p.token_position() - 1);
				std::string asname{ p.lexer().get(token->start(), token->end()) };
				std::static_pointer_cast<Import>(p.stack().back())->set_asname(asname);
			}
			return true;
		}
		return false;
	}
};

struct DottedAsNamesPattern : Pattern<DottedAsNamesPattern>
{
	// dotted_as_names:
	//     | ','.dotted_as_name+
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("dotted_as_names");
		using pattern1 = PatternMatch<OneOrMorePattern<ApplyInBetweenPattern<DottedAsNamePattern,
			SingleTokenPattern<Token::TokenType::COMMA>>>>;
		if (pattern1::match(p)) {
			spdlog::debug("','.dotted_as_name+");
			return true;
		}
		return false;
	}
};

struct ImportNamePattern : Pattern<ImportNamePattern>
{
	// import_name: 'import' dotted_as_names
	static bool matches_impl(Parser &p)
	{
		p.push_to_stack(std::make_shared<Import>());
		spdlog::debug("import_name");
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ImportPattern>,
				DottedAsNamesPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("'import' dotted_as_names");
			return true;
		}
		return false;
	}
};


struct ImportFromPattern : Pattern<ImportFromPattern>
{
	// import_from:
	// | 'from' ('.' | '...')* dotted_name 'import' import_from_targets
	// | 'from' ('.' | '...')+ 'import' import_from_targets
	static bool matches_impl(Parser &)
	{
		spdlog::debug("import_from");
		return false;
	}
};


struct ImportStatementPattern : Pattern<ImportStatementPattern>
{
	// import_stmt:
	// 		| import_name
	//		| import_from
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("import_stmt");
		using pattern1 = PatternMatch<ImportNamePattern>;
		if (pattern1::match(p)) {
			spdlog::debug("import_name");
			return true;
		}
		using pattern2 = PatternMatch<ImportFromPattern>;
		if (pattern2::match(p)) {
			spdlog::debug("import_from");
			return true;
		}
		return false;
	}
};


struct RaiseStatementPattern : Pattern<RaiseStatementPattern>
{
	// raise_stmt:
	//     | 'raise' expression ['from' expression ]
	//     | 'raise'
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("raise_stmt");
		const auto initial_stack_size = p.stack().size();
		using pattern1 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, RaisePattern>,
			ExpressionPattern,
			ZeroOrOnePattern<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, FromPattern>,
				ExpressionPattern>>;
		if (pattern1::match(p)) {
			spdlog::debug("'raise' expression ['from' expression ] ");
			ASSERT((p.stack().size() - initial_stack_size) > 0)
			ASSERT((p.stack().size() - initial_stack_size) < 3)
			if ((p.stack().size() - initial_stack_size) == 1) {
				const auto &exception = p.pop_back();
				p.push_to_stack(std::make_shared<Raise>(exception, nullptr));

			} else {
				const auto cause = p.pop_back();
				const auto &exception = p.pop_back();
				p.push_to_stack(std::make_shared<Raise>(exception, cause));
			}
			return true;
		}

		using pattern2 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, RaisePattern>>;
		if (pattern2::match(p)) {
			spdlog::debug("'raise'");
			ASSERT(p.stack().size() == initial_stack_size)
			p.push_to_stack(std::make_shared<Raise>());
			return true;
		}
		return false;
	}
};

struct AssertStatementPattern : Pattern<AssertStatementPattern>
{
	// assert_stmt: 'assert' expression [',' expression ]
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("AssertStatementPattern");
		const auto initial_stack_size = p.stack().size();
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, AssertPattern>,
				ExpressionPattern,
				ZeroOrMorePattern<SingleTokenPattern<Token::TokenType::COMMA>, ExpressionPattern>>;
		if (pattern1::match(p)) {
			ASSERT((p.stack().size() - initial_stack_size) > 0)
			ASSERT((p.stack().size() - initial_stack_size) <= 2)

			std::shared_ptr<ASTNode> test{ nullptr };
			std::shared_ptr<ASTNode> msg{ nullptr };

			if ((p.stack().size() - initial_stack_size) == 1) {
				test = p.pop_back();
			} else {
				msg = p.pop_back();
				test = p.pop_back();
			}
			p.push_to_stack(std::make_shared<Assert>(test, msg));
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
		using pattern4 = PatternMatch<ImportStatementPattern>;
		if (pattern4::match(p)) {
			spdlog::debug("import_stmt");
			return true;
		}
		using pattern5 = PatternMatch<RaiseStatementPattern>;
		if (pattern5::match(p)) {
			spdlog::debug("raise_stmt");
			return true;
		}
		using pattern9 = PatternMatch<AssertStatementPattern>;
		if (pattern9::match(p)) {
			spdlog::debug("assert_stmt");
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
			auto arg = p.pop_back();
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
			auto arg = p.pop_back();
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
			for (const auto &node : p.stack()) { node->print_node(""); }
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
		BlockScope scope{ p };
		const auto stack_size = p.stack().size();
		// function_def block
		using pattern1 = PatternMatch<FunctionDefinitionPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("function_def_raw: function_def");
			auto name = p.pop_front();
			auto args = [&]() -> std::shared_ptr<ast::ASTNode> {
				if ((p.stack().size() - stack_size) > 0) {
					return p.pop_front();
				} else {
					return std::make_shared<Arguments>();
				}
			}();
			if (args) { args->print_node(""); }
			name->print_node("");
			std::vector<std::shared_ptr<ASTNode>> body;
			{
				BlockScope inner_scope{ p };
				using pattern1a = PatternMatch<ZeroOrOnePattern<BlockPattern>>;
				if (pattern1a::match(p)) { spdlog::debug("block"); }
				for (auto &&node : p.stack()) { body.push_back(std::move(node)); }
			}

			ASSERT(as<Constant>(name));
			ASSERT(as<Arguments>(args));
			auto function = std::make_shared<FunctionDefinition>(
				std::get<String>(as<Constant>(name)->value()).s,
				as<Arguments>(args),
				body,
				std::vector<std::shared_ptr<ASTNode>>{},
				nullptr,
				"");
			scope.parent().push_back(function);
			function->print_node("");
			return true;
		}
		return false;
	}
};


struct FunctionDefinitionStatementPattern : Pattern<FunctionDefinitionStatementPattern>
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

struct IfPattern
{
	static bool matches(std::string_view token_value) { return token_value == "if"; }
};


struct ElifPattern
{
	static bool matches(std::string_view token_value) { return token_value == "elif"; }
};

struct ElsePattern
{
	static bool matches(std::string_view token_value) { return token_value == "else"; }
};


struct ElseBlockStatementPattern : Pattern<ElseBlockStatementPattern>
{
	// else_block: 'else' ':' block
	static bool matches_impl(Parser &p)
	{
		// else_block: 'else' ':' block
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ElsePattern>,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("else_block: 'else' ':' block");
			return true;
		}
		return false;
	}
};


struct ElifStatementPattern : Pattern<ElifStatementPattern>
{
	// elif_stmt:
	//     | 'elif' named_expression ':' block elif_stmt
	//     | 'elif' named_expression ':' block [else_block]
	static bool matches_impl(Parser &p)
	{
		BlockScope scope{ p };
		// 'elif' named_expression ':' block
		using pattern0 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ElifPattern>,
				NamedExpressionPattern,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern0::match(p)) {
			spdlog::debug("'if' named_expression ':' block");
			std::vector<std::shared_ptr<ASTNode>> orelse;
			std::vector<std::shared_ptr<ASTNode>> body;
			{
				BlockScope inner_scope{ p };
				using pattern1 = PatternMatch<ElifStatementPattern>;
				if (pattern1::match(p)) { spdlog::debug("elif_stmt"); }
				using pattern2 = PatternMatch<ZeroOrOnePattern<ElseBlockStatementPattern>>;
				if (pattern2::match(p)) { spdlog::debug("[else_block]"); }
				for (auto &&node : p.stack()) { orelse.push_back(std::move(node)); }
			}
			auto test = p.pop_front();
			while (!p.stack().empty()) { body.push_back(p.pop_front()); }
			scope.parent().push_back(std::make_shared<If>(test, body, orelse));
			return true;
		}
		return false;
	}
};


struct IfStatementPattern : Pattern<IfStatementPattern>
{
	// if_stmt:
	//     | 'if' named_expression ':' block elif_stmt
	//     | 'if' named_expression ':' block [else_block]
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("IfStatementPattern");
		BlockScope scope{ p };

		// 'if' named_expression ':' block
		using pattern0 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, IfPattern>,
				NamedExpressionPattern,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern0::match(p)) {
			spdlog::debug("'if' named_expression ':' block");
			std::vector<std::shared_ptr<ASTNode>> orelse;
			std::vector<std::shared_ptr<ASTNode>> body;
			{
				BlockScope inner_scope{ p };
				using pattern1 = PatternMatch<ElifStatementPattern>;
				if (pattern1::match(p)) { spdlog::debug("elif_stmt"); }
				using pattern2 = PatternMatch<ZeroOrOnePattern<ElseBlockStatementPattern>>;
				if (pattern2::match(p)) { spdlog::debug("[else_block]"); }
				for (auto &&node : p.stack()) { orelse.push_back(std::move(node)); }
			}
			auto test = p.pop_front();
			while (!p.stack().empty()) { body.push_back(p.pop_front()); }
			scope.parent().push_back(std::make_shared<If>(test, body, orelse));
			return true;
		}
		return false;
	}
};


struct ClassPattern
{
	static bool matches(std::string_view token_value) { return token_value == "class"; }
};


struct ClassDefinitionRawPattern : Pattern<ClassDefinitionRawPattern>
{
	// class_def_raw:
	//     | 'class' NAME ['(' [arguments] ')' ] ':' block
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("ClassDefinitionRawPattern");
		BlockScope scope{ p };

		// 'class' NAME ['(' [arguments] ')' ] ':'
		using pattern0 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ClassPattern>,
				SingleTokenPattern<Token::TokenType::NAME>,
				SingleTokenPattern<Token::TokenType::COLON>>;
		if (pattern0::match(p)) {
			spdlog::debug("'class' NAME ['(' [arguments] ')' ] ':'");
			// FIXME: assumes no inheritance
			auto token = p.lexer().peek_token(p.token_position() - 2);
			spdlog::debug("{}", token->to_string());
			std::string class_name{ token->start().pointer_to_program,
				token->end().pointer_to_program };

			std::vector<std::shared_ptr<ASTNode>> body;
			{
				BlockScope block_scope{ p };
				using pattern1 = PatternMatch<BlockPattern>;
				if (pattern1::match(p)) {
					spdlog::debug("block");
				} else {
					return false;
				}
				for (auto &&node : p.stack()) { body.push_back(std::move(node)); }
			}
			std::shared_ptr<Arguments> arguments;
			std::vector<std::shared_ptr<ASTNode>> decorator_list;
			// while (!p.stack().empty()) { arguments.push_back(p.pop_front()); }
			scope.parent().push_back(
				std::make_shared<ClassDefinition>(class_name, arguments, body, decorator_list));
			return true;
		}
		return false;
	}
};

struct ClassDefinitionPattern : Pattern<ClassDefinitionPattern>
{
	// class_def:
	//     | decorators class_def_raw
	//     | class_def_raw
	static bool matches_impl(Parser &p)
	{
		// class_def_raw
		using pattern2 = PatternMatch<ClassDefinitionRawPattern>;
		if (pattern2::match(p)) {
			spdlog::debug("class_def_raw");
			return true;
		}
		return false;
	}
};

struct ForPattern
{
	static bool matches(std::string_view token_value) { return token_value == "for"; }
};

struct InPattern
{
	static bool matches(std::string_view token_value) { return token_value == "in"; }
};

struct ForStatementPattern : Pattern<ForStatementPattern>
{
	// for_stmt:
	//     | 'for' star_targets 'in' ~ star_expressions ':' [TYPE_COMMENT] block [else_block]
	//     | ASYNC 'for' star_targets 'in' ~ star_expressions ':' [TYPE_COMMENT] block [else_block]
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("ForStatementPattern");
		BlockScope scope{ p };
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ForPattern>,
				StarTargetsPattern,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, InPattern>,
				StarExpressionsPattern,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("for_stmt");
			std::vector<std::shared_ptr<ASTNode>> orelse;
			{
				BlockScope else_scope{ p };
				using pattern1 = PatternMatch<ZeroOrOnePattern<ElseBlockStatementPattern>>;
				if (pattern1::match(p)) { spdlog::debug("[else_block]"); }
				for (auto &&node : p.stack()) { orelse.push_back(std::move(node)); }
			}
			// FIXME: type comment is currently not considered
			std::string type_comment{ "" };
			auto target = p.pop_front();
			auto iter = p.pop_front();
			std::vector<std::shared_ptr<ASTNode>> body;
			while (!p.stack().empty()) { body.push_back(p.pop_front()); }
			scope.parent().push_back(
				std::make_shared<For>(target, iter, body, orelse, type_comment));
			return true;
		}

		return false;
	}
};

struct ExceptBlockPattern : Pattern<ExceptBlockPattern>
{
	// except_block:
	//     | 'except' expression ['as' NAME ] ':' block
	//     | 'except' ':' block
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("ExceptBlockPattern");
		{
			BlockScope scope{ p };
			using pattern1 =
				PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ExceptPattern>,
					ExpressionPattern>;
			if (pattern1::match(p)) {
				spdlog::debug("'except' expression");
				std::string name{};
				using pattern1a =
					PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, AsPattern>,
						SingleTokenPattern<Token::TokenType::NAME>>;
				if (pattern1a::match(p)) {
					spdlog::debug("['as' NAME ]");
					auto token = p.lexer().peek_token(p.token_position() - 1);
					spdlog::debug("{}", token->to_string());
					name = std::string{ token->start().pointer_to_program,
						token->end().pointer_to_program };
				}
				using pattern1b =
					PatternMatch<SingleTokenPattern<Token::TokenType::COLON>, BlockPattern>;
				if (pattern1b::match(p)) {
					spdlog::debug("':' block");
					std::vector<std::shared_ptr<ASTNode>> body;
					const auto type = p.pop_front();
					while (!p.stack().empty()) { body.push_back(p.pop_front()); }
					scope.parent().push_front(std::make_shared<ExceptHandler>(type, name, body));
					return true;
				}
			}
		}
		{
			BlockScope scope{ p };
			using pattern2 =
				PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ExceptPattern>,
					SingleTokenPattern<Token::TokenType::COLON>,
					BlockPattern>;
			if (pattern2::match(p)) {
				spdlog::debug("'except' ':' block");
				std::vector<std::shared_ptr<ASTNode>> body;
				while (!p.stack().empty()) { body.push_back(p.pop_front()); }
				scope.parent().push_front(std::make_shared<ExceptHandler>(nullptr, "", body));
				return true;
			}
		}
		return false;
	}
};

struct FinallyBlockPattern : Pattern<FinallyBlockPattern>
{
	// finally_block:
	//     | 'finally' ':' block
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("FinallyBlockPattern");
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, FinallyPattern>,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("'finally' ':' block");
			return true;
		}
		return false;
	}
};

struct TryStatementPattern : Pattern<TryStatementPattern>
{
	// try_stmt:
	// 		| 'try' ':' block finally_block
	// 		| 'try' ':' block except_block+ [else_block] [finally_block]
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("TryStatementPattern");
		BlockScope scope{ p };
		// 'try' ':' block
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, TryPattern>,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("'try' ':' block");

			std::vector<std::shared_ptr<ASTNode>> body;
			std::vector<std::shared_ptr<ExceptHandler>> handlers;
			std::vector<std::shared_ptr<ASTNode>> orelse;
			std::vector<std::shared_ptr<ASTNode>> finally;
			bool match = false;
			// finally_block
			{
				BlockScope finally_scope{ p };
				using pattern1a = PatternMatch<FinallyBlockPattern>;
				if (pattern1a::match(p)) {
					spdlog::debug("finally_block");
					while (!p.stack().empty()) { finally.push_back(p.pop_front()); }
					match = true;
				}
			}
			// except_block+
			{
				BlockScope except_block{ p };
				using pattern1b = PatternMatch<OneOrMorePattern<ExceptBlockPattern>>;
				if (pattern1b::match(p)) {
					spdlog::debug("except_block");
					while (!p.stack().empty()) {
						auto node = p.pop_back();
						ASSERT(as<ExceptHandler>(node))
						handlers.push_back(as<ExceptHandler>(node));
					}
					match = true;

					// [else_block]
					{
						BlockScope else_block{ p };
						using pattern1c = PatternMatch<ElseBlockStatementPattern>;
						if (pattern1c::match(p)) {
							spdlog::debug("[else_block]");
							while (!p.stack().empty()) { orelse.push_back(p.pop_front()); }
						}
					}
					// [finally_block]
					{
						BlockScope finally_block{ p };
						using pattern1d = PatternMatch<FinallyBlockPattern>;
						if (pattern1d::match(p)) {
							spdlog::debug("[finally_block]");
							while (!p.stack().empty()) { finally.push_back(p.pop_front()); }
						}
					}
				}
			}
			if (!match) { return false; }
			while (!p.stack().empty()) { body.push_back(p.pop_front()); }
			scope.parent().push_back(std::make_shared<Try>(body, handlers, orelse, finally));
			return true;
		}
		return false;
	}
};


struct WhileStatementPattern : Pattern<WhileStatementPattern>
{
	// while_stmt:
	//     | 'while' named_expression ':' block [else_block]
	static bool matches_impl(Parser &p)
	{
		spdlog::debug("WhileStatementPattern");
		BlockScope scope{ p };

		// 'while' named_expression ':' block
		using pattern0 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, WhilePattern>,
				NamedExpressionPattern,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern0::match(p)) {
			spdlog::debug("'while' named_expression ':' block");
			std::vector<std::shared_ptr<ASTNode>> orelse;
			std::vector<std::shared_ptr<ASTNode>> body;
			{
				BlockScope inner_scope{ p };
				using pattern1 = PatternMatch<ZeroOrOnePattern<ElseBlockStatementPattern>>;
				if (pattern1::match(p)) { spdlog::debug("[else_block]"); }
				for (auto &&node : p.stack()) { orelse.push_back(std::move(node)); }
			}
			auto test = p.pop_front();
			while (!p.stack().empty()) { body.push_back(p.pop_front()); }
			scope.parent().push_back(std::make_shared<While>(test, body, orelse));
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
		using pattern1 = PatternMatch<FunctionDefinitionStatementPattern>;
		if (pattern1::match(p)) {
			spdlog::debug("function_def");
			p.print_stack();
			return true;
		}
		// if_stmt
		using pattern2 = PatternMatch<IfStatementPattern>;
		if (pattern2::match(p)) {
			spdlog::debug("if_stmt");
			p.print_stack();
			return true;
		}

		using pattern3 = PatternMatch<ClassDefinitionPattern>;
		if (pattern3::match(p)) {
			spdlog::debug("class_def");
			p.print_stack();
			return true;
		}

		// for_stmt
		using pattern5 = PatternMatch<ForStatementPattern>;
		if (pattern5::match(p)) {
			spdlog::debug("for_stmt");
			p.print_stack();
			return true;
		}

		// try_stmt
		using pattern6 = PatternMatch<TryStatementPattern>;
		if (pattern6::match(p)) {
			spdlog::debug("try_stmt");
			p.print_stack();
			return true;
		}

		// while_stmt
		using pattern7 = PatternMatch<WhileStatementPattern>;
		if (pattern7::match(p)) {
			spdlog::debug("while_stmt");
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
		while (pattern1::match(p)) { p.commit(); }
		return true;
	}
};

struct FilePattern : Pattern<FilePattern>
{
	// file: [statements] ENDMARKER
	static bool matches_impl(Parser &p)
	{
		BlockScope scope{ p };
		using pattern1 =
			PatternMatch<StatementsPattern, SingleTokenPattern<Token::TokenType::ENDMARKER>>;
		if (pattern1::match(p)) {
			for (auto &&node : p.stack()) { p.module()->emplace(std::move(node)); }
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