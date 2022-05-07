#include "Parser.hpp"
#include "runtime/Value.hpp"
#include <sstream>

using namespace py;

#define PARSER_ERROR()                                           \
	do {                                                         \
		spdlog::error("Parser error {}:{}", __FILE__, __LINE__); \
		std::abort();                                            \
	} while (0);

// #ifndef NDEBUG
// #define DEBUG_LOG(...)              \
// 	do {                            \
// 		spdlog::trace(__VA_ARGS__); \
// 	} while (0);
// #else
#define DEBUG_LOG(MSG, ...)
// #endif

// #ifndef NDEBUG
// #define PRINT_STACK() p.print_stack();
// #else
#define PRINT_STACK()
// #endif

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
			// std::string str;
			// for (size_t i = original_token_position; i < p.token_position(); ++i) {
			// 	str += std::string(p.lexer().peek_token(i)->start().pointer_to_program,
			// 		p.lexer().peek_token(i)->end().pointer_to_program);
			// 	str += ' ';
			// }
			// DEBUG_LOG("match: " + str + '\n');
			if constexpr (has_advance_by<CurrentType>::value) {
				p.token_position() += CurrentType::advance_by;
			}
			if constexpr (TypeIdx + 1 == std::tuple_size_v<PatternTuple>) {
				return true;
			} else {
				return PatternMatch_<TypeIdx + 1, PatternTuple>::match(p);
			}
		} else {
			// std::string str;
			// for (size_t i = original_token_position; i < p.token_position(); ++i) {
			// 	str += std::string(p.lexer().peek_token(i)->start().pointer_to_program,
			// 		p.lexer().peek_token(i)->end().pointer_to_program);
			// 	str += ' ';
			// }
			// DEBUG_LOG("no match: " + str + '\n');
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
template<typename MaInKeywordPatternType, typename InBetweenPattern>
struct ApplyInBetweenPattern
	: Pattern<ApplyInBetweenPattern<MaInKeywordPatternType, InBetweenPattern>>
{
	static bool matches_impl(Parser &p)
	{
		using MaInKeywordPatternType_ = PatternMatch<MaInKeywordPatternType>;
		using InBetweenPattern_ = PatternMatch<InBetweenPattern>;

		if (!MaInKeywordPatternType_::match(p)) { return false; }

		auto original_token_position = p.token_position();
		while (InBetweenPattern_::match(p)) {
			if (!MaInKeywordPatternType_::match(p)) {
				p.token_position() = original_token_position;
				break;
			}
			original_token_position = p.token_position();
		}
		DEBUG_LOG(
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
		DEBUG_LOG("ZeroOrOnePattern (no match): {}",
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
		const auto initial_position = p.stack().size();
		bool is_match = false;
		{
			BlockScope scope{ p };
			is_match = PatternType::matches(p);
		}
		p.token_position() = start_position;
		while (initial_position > p.stack().size()) { p.pop_back(); }
		return is_match;
	}
};


template<typename... Ts> struct is_tuple : std::false_type
{
};

template<typename... Ts> struct is_tuple<std::tuple<Ts...>> : std::true_type
{
};

template<typename T> static constexpr bool is_tuple_v = is_tuple<T>{};

template<typename lhs, typename rhs> struct AndLiteral : Pattern<AndLiteral<lhs, rhs>>
{
	static constexpr size_t advance_by = lhs::advance_by;

	template<typename PatternType>
	static bool matches_(std::string_view token) requires(!is_tuple_v<PatternType>)
	{
		return PatternType::matches(token);
	}

	template<typename PatternTypes>
	static bool matches_(std::string_view token) requires(is_tuple_v<PatternTypes>)
	{
		return std::apply(
			[&](auto... r) { return (... && decltype(r)::matches(token)); }, PatternTypes{});
	}

	static bool matches_impl(Parser &p)
	{
		if (lhs::matches(p)) {
			const auto token = p.lexer().peek_token(p.token_position());
			const size_t size =
				std::distance(token->start().pointer_to_program, token->end().pointer_to_program);
			std::string_view token_sv{ token->start().pointer_to_program, size };
			return matches_<rhs>(token_sv);
		}
		return false;
	}
};


template<typename lhs, typename rhs> struct AndNotLiteral : Pattern<AndNotLiteral<lhs, rhs>>
{
	static constexpr size_t advance_by = lhs::advance_by;

	template<typename PatternType>
	static bool matches_(std::string_view token) requires(!is_tuple_v<PatternType>)
	{
		return !PatternType::matches(token);
	}

	template<typename PatternTypes>
	static bool matches_(std::string_view token) requires(is_tuple_v<PatternTypes>)
	{
		return !std::apply(
			[&](auto... r) { return (... || decltype(r)::matches(token)); }, PatternTypes{});
	}

	static bool matches_impl(Parser &p)
	{
		if (lhs::matches(p)) {
			const auto token = p.lexer().peek_token(p.token_position());
			const size_t size =
				std::distance(token->start().pointer_to_program, token->end().pointer_to_program);
			std::string_view token_sv{ token->start().pointer_to_program, size };
			return matches_<rhs>(token_sv);
		}
		return true;
	}
};

struct AndKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "and"; }
};

struct AsKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "as"; }
};

struct AssertKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "assert"; }
};

struct BreakKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "break"; }
};

struct ContinueKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "continue"; }
};

struct DeleteKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "del"; }
};


struct ElifKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "elif"; }
};

struct ElseKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "else"; }
};

struct ExceptKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "except"; }
};

struct FinallyKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "finally"; }
};

struct ForKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "for"; }
};

struct FromKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "from"; }
};

struct GlobalKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "global"; }
};

struct IfKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "if"; }
};

struct IsKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "is"; }
};

struct ImportKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "import"; }
};

struct InKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "in"; }
};

struct NotKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "not"; }
};

struct OrKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "or"; }
};

struct PassKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "pass"; }
};

struct RaiseKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "raise"; }
};

struct ReturnPattern
{
	static bool matches(std::string_view token_value) { return token_value == "return"; }
};

struct TryKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "try"; }
};

struct WhileKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "while"; }
};

struct WithKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "with"; }
};

using ReservedKeywords = std::tuple<AndKeywordPattern,
	AsKeywordPattern,
	AssertKeywordPattern,
	BreakKeywordPattern,
	ContinueKeywordPattern,
	DeleteKeywordPattern,
	ElifKeywordPattern,
	ElseKeywordPattern,
	ExceptKeywordPattern,
	FinallyKeywordPattern,
	ForKeywordPattern,
	FromKeywordPattern,
	GlobalKeywordPattern,
	IfKeywordPattern,
	IsKeywordPattern,
	ImportKeywordPattern,
	InKeywordPattern,
	NotKeywordPattern,
	OrKeywordPattern,
	PassKeywordPattern,
	RaiseKeywordPattern,
	ReturnPattern,
	TryKeywordPattern,
	WhileKeywordPattern,
	WithKeywordPattern>;

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
			DEBUG_LOG("'(' target_with_star_atom ')'");

			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string id{ p.lexer().get(token->start(), token->end()) };
			p.push_to_stack(std::make_shared<Name>(
				id, ContextType::STORE, SourceLocation{ token->start(), token->end() }));
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
		DEBUG_LOG("t_lookahead");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());

		using pattern1 = PatternMatch<OrPattern<SingleTokenPattern<Token::TokenType::LPAREN>,
			SingleTokenPattern<Token::TokenType::LSQB>,
			SingleTokenPattern<Token::TokenType::DOT>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("t_lookahead: '(' | '[' | '.'");
			return true;
		}
		return false;
	}
};

struct AtomPattern;
struct SlicesPattern;
struct ArgumentsPattern;

struct TPrimaryPattern_ : Pattern<TPrimaryPattern_>
{
	// t_primary' | '.' NAME &t_lookahead t_primary'
	//            | '[' slices ']' &t_lookahead t_primary'
	//            | genexp &t_lookahead t_primary'
	//            | '(' [arguments] ')' &t_lookahead t_primary'
	//            | ϵ
	static bool matches_impl(Parser &p)
	{
		auto token = p.lexer().peek_token(p.token_position() + 1);
		DEBUG_LOG("TPrimaryPattern_")
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::DOT>,
			SingleTokenPattern<Token::TokenType::NAME>,
			LookAhead<TLookahead>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'.' NAME &t_lookahead")
			std::string_view name{ token->start().pointer_to_program,
				token->end().pointer_to_program };
			auto value = p.pop_back();
			p.push_to_stack(std::make_shared<Attribute>(value,
				std::string(name),
				ContextType::LOAD,
				SourceLocation{ token->start(), token->end() }));
			if (PatternMatch<TPrimaryPattern_>::match(p)) {
				DEBUG_LOG("'.' NAME &t_lookahead t_primary'")
				return true;
			}
		}

		using pattern2 = PatternMatch<SingleTokenPattern<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPattern<Token::TokenType::RSQB>,
			LookAhead<TLookahead>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("'[' slices ']' &t_lookahead")
			auto subscript = p.pop_back();
			auto value = p.pop_back();

			ASSERT(as<Subscript>(subscript))
			as<Subscript>(subscript)->set_value(value);
			as<Subscript>(subscript)->set_context(ContextType::LOAD);
			p.push_to_stack(subscript);
			if (PatternMatch<TPrimaryPattern_>::match(p)) {
				DEBUG_LOG("'[' slices ']' &t_lookahead t_primary'")
				return true;
			}
		}

		{
			BlockScope scope{ p };
			using pattern4 = PatternMatch<SingleTokenPattern<Token::TokenType::LPAREN>,
				ZeroOrOnePattern<ArgumentsPattern>,
				SingleTokenPattern<Token::TokenType::RPAREN>,
				LookAhead<TLookahead>>;
			if (pattern4::match(p)) {
				DEBUG_LOG("'(' [arguments] ')' &t_lookahead")
				const auto &caller = scope.parent().back();
				scope.parent().pop_back();
				std::vector<std::shared_ptr<ASTNode>> args;
				std::vector<std::shared_ptr<Keyword>> kwargs;
				while (!p.stack().empty()) {
					auto node = p.pop_front();
					if (auto keyword_node = as<Keyword>(node)) {
						kwargs.push_back(keyword_node);
					} else {
						args.push_back(node);
					}
				}
				auto end_token = p.lexer().peek_token(p.token_position() - 1);
				p.push_to_stack(std::make_shared<Call>(
					caller, args, kwargs, SourceLocation{ token->start(), end_token->end() }));
				if (PatternMatch<TPrimaryPattern_>::match(p)) {
					DEBUG_LOG("'(' [arguments] ')' &t_lookahead t_primary'")
					scope.parent().push_back(p.pop_back());
					return true;
				}
			}
		}
		using pattern5 = PatternMatch<LookAhead<OrPattern<AtomPattern, TLookahead>>>;
		if (pattern5::match(p)) {
			DEBUG_LOG("t_primary' | ϵ")
			return true;
		}
		return false;
	}
};

struct TPrimaryPattern : Pattern<TPrimaryPattern>
{
	// t_primary:
	// 	| t_primary '.' NAME &t_lookahead
	// 	| t_primary '[' slices ']' &t_lookahead
	// 	| t_primary genexp &t_lookahead
	// 	| t_primary '(' [arguments] ')' &t_lookahead
	// 	| atom &t_lookahead

	// t_primary | atom &t_lookahead t_primary'

	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("t_primary");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());

		using pattern1 = PatternMatch<AtomPattern, LookAhead<TLookahead>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("atom &t_lookahead'");
			if (PatternMatch<TPrimaryPattern_>::match(p)) {
				DEBUG_LOG("atom &t_lookahead t_primary'");
				return true;
			}
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
		DEBUG_LOG("target_with_star_atom");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());

		// t_primary '.' NAME !t_lookahead
		using pattern1 = PatternMatch<TPrimaryPattern,
			SingleTokenPattern<Token::TokenType::DOT>,
			SingleTokenPattern<Token::TokenType::NAME>,
			NegativeLookAhead<TLookahead>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("t_primary '.' NAME !t_lookahead");
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			const auto primary = p.pop_back();
			DEBUG_LOG("{}", name);
			primary->print_node("");
			auto attribute = std::make_shared<Attribute>(
				primary, name, ContextType::STORE, SourceLocation{ token->start(), token->end() });
			p.push_to_stack(attribute);
			return true;
		}

		using pattern2 = PatternMatch<TPrimaryPattern,
			SingleTokenPattern<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPattern<Token::TokenType::RSQB>,
			NegativeLookAhead<TLookahead>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("t_primary '[' slices ']' !t_lookahead");
			auto subscript = p.pop_back();
			auto value = p.pop_back();
			ASSERT(as<Subscript>(subscript))
			as<Subscript>(subscript)->set_value(value);
			as<Subscript>(subscript)->set_context(ContextType::STORE);
			p.push_to_stack(subscript);
			return true;
		}

		// star_atom
		using pattern3 = PatternMatch<StarAtomPattern>;
		if (pattern3::match(p)) {
			DEBUG_LOG("star_atom");
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
		DEBUG_LOG("star_target");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());

		// target_with_star_atom
		using pattern2 = PatternMatch<TargetWithStarAtomPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("target_with_star_atom");
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
		BlockScope scope{ p };
		DEBUG_LOG("star_targets");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());

		// star_target !','
		using pattern1 = PatternMatch<StarTargetPattern,
			NegativeLookAhead<SingleTokenPattern<Token::TokenType::COMMA>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("star_target !','");
			ASSERT(p.stack().size() == 1);
			scope.parent().push_back(p.pop_back());
			return true;
		}
		using pattern2 = PatternMatch<StarTargetPattern,
			OneOrMorePattern<SingleTokenPattern<Token::TokenType::COMMA>, StarTargetPattern>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("star_target (',' star_target )* [',']");
			ASSERT(p.stack().size() >= 1);
			std::vector<std::shared_ptr<ASTNode>> nodes;
			nodes.reserve(p.stack().size());
			while (!p.stack().empty()) { nodes.push_back(p.pop_front()); }
			scope.parent().push_back(std::make_shared<Tuple>(nodes,
				ContextType::STORE,
				SourceLocation{
					nodes.front()->source_location().start, nodes.back()->source_location().end }));
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
		const auto initial_token_position = p.token_position();
		using pattern1 =
			PatternMatch<OneOrMorePattern<SingleTokenPattern<Token::TokenType::STRING>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("strings: STRING+");
			std::string complete_string;
			auto start_token = p.lexer().peek_token(initial_token_position);
			auto end_token = p.lexer().peek_token(initial_token_position);
			for (size_t idx = initial_token_position; idx < p.token_position(); ++idx) {
				auto token = p.lexer().peek_token(idx);
				auto is_triple_quote = [token]() {
					const char *ptr = token->start().pointer_to_program;
					return (ptr[0] == '\"' || ptr[0] == '\'') && (ptr[1] == '\"' || ptr[1] == '\'')
						   && (ptr[2] == '\"' || ptr[2] == '\'');
				};
				const auto value = [&token, &is_triple_quote]() {
					if (is_triple_quote()) {
						return std::string{ token->start().pointer_to_program + 3,
							token->end().pointer_to_program - 3 };
					} else {
						return std::string{ token->start().pointer_to_program + 1,
							token->end().pointer_to_program - 1 };
					}
				}();
				complete_string += value;
				end_token = token;
			}
			p.push_to_stack(std::make_shared<Constant>(
				complete_string, SourceLocation{ start_token->start(), end_token->end() }));
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
		DEBUG_LOG("star_named_expressions");
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
		DEBUG_LOG("list");
		auto start_token = p.lexer().peek_token(p.token_position());
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::LSQB>,
			ZeroOrMorePattern<StarNamedExpressions>,
			SingleTokenPattern<Token::TokenType::RSQB>>;
		if (pattern1::match(p)) {
			std::vector<std::shared_ptr<ASTNode>> elements;
			elements.reserve(p.stack().size());
			while (!p.stack().empty()) { elements.push_back(p.pop_front()); }

			auto end_token = p.lexer().peek_token(p.token_position() - 1);
			auto list = std::make_shared<List>(elements,
				ContextType::LOAD,
				SourceLocation{ start_token->start(), end_token->end() });

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
		BlockScope tuple_scope{ p };
		DEBUG_LOG("tuple");
		auto start_token = p.lexer().peek_token(p.token_position());
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::LPAREN>,
			ZeroOrMorePattern<StarNamedExpression,
				SingleTokenPattern<Token::TokenType::COMMA>,
				ZeroOrMorePattern<StarNamedExpressions>>,
			SingleTokenPattern<Token::TokenType::RPAREN>>;
		if (pattern1::match(p)) {
			std::vector<std::shared_ptr<ASTNode>> elements;
			elements.reserve(p.stack().size());
			while (!p.stack().empty()) { elements.push_back(p.pop_front()); }
			auto end_token = p.lexer().peek_token(p.token_position() - 1);
			auto tuple = std::make_shared<Tuple>(elements,
				ContextType::LOAD,
				SourceLocation{ start_token->start(), end_token->end() });
			tuple_scope.parent().push_back(std::move(tuple));
			return true;
		}
		return false;
	}
};


struct GroupPattern : Pattern<GroupPattern>
{
	static bool matches_impl(Parser &p)
	{
		// group:
		//     | '(' (yield_expr | named_expression) ')'
		DEBUG_LOG("group");

		// TODO: add yield_expr
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::LPAREN>,
			NamedExpressionPattern,
			SingleTokenPattern<Token::TokenType::RPAREN>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'(' (yield_expr | named_expression) ')'");
			return true;
		}
		return false;
	}
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
		DEBUG_LOG("kvpair");
		using pattern1 = PatternMatch<ExpressionPattern,
			SingleTokenPattern<Token::TokenType::COLON>,
			ExpressionPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("kvpair: expression ':' expression");
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
		DEBUG_LOG("double_starred_kvpair");
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::DOUBLESTAR>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'**' bitwise_or");
			return true;
		}

		using pattern2 = PatternMatch<KVPairPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("kvpair");
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
		DEBUG_LOG("double_starred_kvpairs");
		using pattern1 = PatternMatch<ApplyInBetweenPattern<DoubleStarredKVPairPattern,
										  SingleTokenPattern<Token::TokenType::COMMA>>,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("','.double_starred_kvpair+ [',']");
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
		// '{' [double_starred_kvpairs] '}'

		auto start_token = p.lexer().peek_token(p.token_position());
		DEBUG_LOG("'{' [double_starred_kvpairs] '}'");
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::LBRACE>,
			ZeroOrOnePattern<DoubleStarredKVPairsPattern>,
			SingleTokenPattern<Token::TokenType::RBRACE>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'{' [double_starred_kvpairs] '}'");
			std::vector<std::shared_ptr<ASTNode>> keys;
			std::vector<std::shared_ptr<ASTNode>> values;

			ASSERT(p.stack().size() % 2 == 0)
			keys.reserve(p.stack().size() / 2);
			values.reserve(p.stack().size() / 2);

			while (!p.stack().empty()) {
				keys.push_back(p.pop_front());
				values.push_back(p.pop_front());
			}

			auto end_token = p.lexer().peek_token(p.token_position() - 1);
			auto dict = std::make_shared<Dict>(
				keys, values, SourceLocation{ start_token->start(), end_token->end() });
			dict_scope.parent().push_back(std::move(dict));
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


// this function assumes that we have a valid number
std::shared_ptr<ASTNode> parse_number(std::string value, SourceLocation source_location)
{
	std::erase_if(value, [](const char c) { return c == '_'; });

	// FIXME: check for overlflow without throwing exceptions
	// TODO:  handle very large ints

	if (value[1] == 'o' || value[1] == 'O') {
		// octal
		std::string oct_str{ value.begin() + 2, value.end() };
		int64_t int_value = std::stoll(oct_str, nullptr, 8);
		return std::make_shared<Constant>(int_value, source_location);
	} else if (value[1] == 'x' || value[1] == 'X') {
		// hex
		int64_t int_value = std::stoll(value, nullptr, 16);
		return std::make_shared<Constant>(int_value, source_location);
	} else if (value[1] == 'b' || value[1] == 'B') {
		// binary
		std::string bin_str{ value.begin() + 2, value.end() };
		int64_t int_value = std::stoll(bin_str, nullptr, 2);
		return std::make_shared<Constant>(int_value, source_location);
	} else if (value.find_first_of("jJ") != std::string::npos) {
		// imaginary number
		TODO();
	} else if (value.find_first_of("eE") != std::string::npos) {
		// scientific notation
		// FIXME: seems innefficient, since we know that it is in scientific notation?
		std::istringstream os(value);
		double float_value;
		os >> float_value;
		return std::make_shared<Constant>(float_value, source_location);
	} else if (value.find('.') != std::string::npos) {
		// float
		double float_value = std::stod(value);
		return std::make_shared<Constant>(float_value, source_location);
	} else {
		// int
		int64_t int_value = std::stoll(value);
		return std::make_shared<Constant>(int_value, source_location);
	}
}


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
		DEBUG_LOG("atom");
		{
			const auto token = p.lexer().peek_token(p.token_position());
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			DEBUG_LOG(name);
		}
		// NAME
		// using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>>;
		using pattern1 = PatternMatch<AndPattern<SingleTokenPattern<Token::TokenType::NAME>,
			AndNotLiteral<SingleTokenPattern<Token::TokenType::NAME>, ReservedKeywords>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("NAME");

			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			if (name == "True") {
				p.push_to_stack(std::make_shared<Constant>(
					true, SourceLocation{ token->start(), token->end() }));
			} else if (name == "False") {
				p.push_to_stack(std::make_shared<Constant>(
					false, SourceLocation{ token->start(), token->end() }));
			} else if (name == "None") {
				p.push_to_stack(std::make_shared<Constant>(py::NameConstant{ py::NoneType{} },
					SourceLocation{ token->start(), token->end() }));
			} else {
				p.push_to_stack(std::make_shared<Name>(
					name, ContextType::LOAD, SourceLocation{ token->start(), token->end() }));
			}
			return true;
		}
		// strings
		using pattern6 = PatternMatch<OneOrMorePattern<StringPattern>>;
		if (pattern6::match(p)) {
			DEBUG_LOG("strings");
			return true;
		}

		// NUMBER
		using pattern7 = PatternMatch<SingleTokenPattern<Token::TokenType::NUMBER>>;
		if (pattern7::match(p)) {
			DEBUG_LOG("NUMBER");
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string number{ token->start().pointer_to_program,
				token->end().pointer_to_program };
			p.push_to_stack(parse_number(number, SourceLocation{ token->start(), token->end() }));

			return true;
		}

		// 	| (tuple | group | genexp)
		using pattern8 = PatternMatch<OrPattern<TuplePattern, GroupPattern, GenexPattern>>;
		if (pattern8::match(p)) {
			DEBUG_LOG("(tuple | group | genexp)");
			return true;
		}

		// 	| (list | listcomp)
		using pattern9 = PatternMatch<OrPattern<ListPattern, ListCompPattern>>;
		if (pattern9::match(p)) {
			DEBUG_LOG("(list | listcomp)");
			return true;
		}

		// (dict | set | dictcomp | setcomp)
		using pattern10 =
			PatternMatch<OrPattern<DictPattern, SetPattern, DictCompPattern, SetCompPattern>>;
		if (pattern10::match(p)) {
			DEBUG_LOG("(dict | set | dictcomp | setcomp)");
			return true;
		}

		using pattern11 = PatternMatch<SingleTokenPattern<Token::TokenType::ELLIPSIS>>;
		if (pattern11::match(p)) {
			auto token = p.lexer().peek_token(p.token_position() - 1);
			p.push_to_stack(std::make_shared<Constant>(
				Ellipsis{}, SourceLocation{ token->start(), token->end() }));
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
		DEBUG_LOG("NamedExpressionPattern");
		const auto token = p.lexer().peek_token(p.token_position());
		DEBUG_LOG("{}", token->to_string());
		std::string_view maybe_name{ token->start().pointer_to_program,
			token->end().pointer_to_program };

		// NAME ':=' ~ expression
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>,
			SingleTokenPattern<Token::TokenType::COLONEQUAL>,
			ExpressionPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("NAME ':=' ~ expression");
			const std::string name{ maybe_name };
			auto target = std::make_shared<Name>(
				name, ContextType::STORE, SourceLocation{ token->start(), token->end() });
			const auto &value = p.pop_back();
			const auto end_token = p.lexer().peek_token(p.token_position());
			p.push_to_stack(std::make_shared<NamedExpr>(
				target, value, SourceLocation{ token->start(), end_token->end() }));
			return true;
		}

		// expression !':='
		using pattern2 = PatternMatch<ExpressionPattern,
			NegativeLookAhead<SingleTokenPattern<Token::TokenType::COLONEQUAL>>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("expression !':='");
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
		DEBUG_LOG("KwargsOrStarredPattern");
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>,
			SingleTokenPattern<Token::TokenType::EQUAL>,
			ExpressionPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("NAME '=' expression");
			const auto &expression = p.pop_back();
			const auto end_token = p.lexer().peek_token(p.token_position());
			p.push_to_stack(std::make_shared<Keyword>(
				maybe_name, expression, SourceLocation{ token->start(), end_token->end() }));
			return true;
		}
		return false;
	}
};

struct StarredExpressionPattern : Pattern<StarredExpressionPattern>
{
	// starred_expression:
	//     | '*' expression
	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("StarredExpressionPattern");
		auto start_token = p.lexer().peek_token(p.token_position());
		// '*' expression
		using pattern1 =
			PatternMatch<SingleTokenPattern<Token::TokenType::STAR>, ExpressionPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'*' expression");
			const auto &arg = p.pop_back();
			const auto end_token = p.lexer().peek_token(p.token_position());
			p.push_to_stack(std::make_shared<Starred>(
				arg, ContextType::LOAD, SourceLocation{ start_token->start(), end_token->end() }));
			return true;
		}
		return false;
	}
};

struct KwargsOrDoubleStarredPattern : Pattern<KwargsOrDoubleStarredPattern>
{
	// kwarg_or_double_starred:
	//     | NAME '=' expression
	//     | '**' expression
	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("KwargsOrDoubleStarredPattern")
		auto token = p.lexer().peek_token(p.token_position());
		std::string_view maybe_name{ token->start().pointer_to_program,
			token->end().pointer_to_program };

		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>,
			SingleTokenPattern<Token::TokenType::EQUAL>,
			ExpressionPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("NAME '=' expression")
			auto expression = p.pop_back();
			const auto end_token = p.lexer().peek_token(p.token_position());
			p.push_to_stack(std::make_shared<Keyword>(std::string(maybe_name),
				expression,
				SourceLocation{ token->start(), end_token->end() }));
			return true;
		}

		using pattern2 =
			PatternMatch<SingleTokenPattern<Token::TokenType::DOUBLESTAR>, ExpressionPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("'**' expression")
			auto expression = p.pop_back();
			const auto end_token = p.lexer().peek_token(p.token_position());
			p.push_to_stack(std::make_shared<Keyword>(
				expression, SourceLocation{ token->start(), end_token->end() }));
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
		DEBUG_LOG("KwargsPattern")

		// ','.kwarg_or_starred+ ',' ','.kwarg_or_double_starred+
		using pattern1 = PatternMatch<OneOrMorePattern<ApplyInBetweenPattern<KwargsOrStarredPattern,
										  SingleTokenPattern<Token::TokenType::COMMA>>>,
			SingleTokenPattern<Token::TokenType::COMMA>,
			OneOrMorePattern<ApplyInBetweenPattern<KwargsOrDoubleStarredPattern,
				SingleTokenPattern<Token::TokenType::COMMA>>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("','.kwarg_or_starred+ ',' ','.kwarg_or_double_starred+")
			return true;
		}

		// ','.kwarg_or_starred+
		using pattern2 = PatternMatch<OneOrMorePattern<ApplyInBetweenPattern<KwargsOrStarredPattern,
			SingleTokenPattern<Token::TokenType::COMMA>>>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("','.kwarg_or_starred+")
			return true;
		}

		// ','.kwarg_or_double_starred+
		using pattern3 =
			PatternMatch<OneOrMorePattern<ApplyInBetweenPattern<KwargsOrDoubleStarredPattern,
				SingleTokenPattern<Token::TokenType::COMMA>>>>;
		if (pattern3::match(p)) {
			DEBUG_LOG("','.kwarg_or_double_starred+")
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
		DEBUG_LOG("ArgsPattern");
		DEBUG_LOG(
			"Testing pattern: ','.(starred_expression | named_expression !'=')+ [',' kwargs ]");
		using pattern1 = PatternMatch<
			ApplyInBetweenPattern<
				OrPattern<StarredExpressionPattern,
					GroupPatterns<NamedExpressionPattern,
						NegativeLookAhead<SingleTokenPattern<Token::TokenType::EQUAL>>>>,
				SingleTokenPattern<Token::TokenType::COMMA>>,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>, KwargsPattern>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("','.(starred_expression | named_expression !'=')+ [',' kwargs ]'");
			DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
			return true;
		}
		using pattern2 = PatternMatch<KwargsPattern>;
		DEBUG_LOG("Testing pattern: kwargs");
		if (pattern2::match(p)) {
			DEBUG_LOG("kwargs");
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
		DEBUG_LOG("ArgumentsPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		using pattern1 = PatternMatch<ArgsPattern,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>>,
			LookAhead<SingleTokenPattern<Token::TokenType::RPAREN>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("args [','] &')'");
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
		DEBUG_LOG("SlicePattern");
		auto start_token = p.lexer().peek_token(p.token_position());
		using pattern1 = PatternMatch<ZeroOrOnePattern<ExpressionPattern>,
			SingleTokenPattern<Token::TokenType::COLON>,
			ZeroOrOnePattern<ExpressionPattern>,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COLON>,
				ZeroOrOnePattern<ExpressionPattern>>>;
		const auto initial_size = p.stack().size();
		if (pattern1::match(p)) {
			DEBUG_LOG("[expression] ':' [expression] [':' [expression] ]");
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
			auto end_token = p.lexer().peek_token(p.token_position() - 1);
			p.push_to_stack(std::make_shared<Subscript>(
				SourceLocation{ start_token->start(), end_token->end() }));
			as<ast::Subscript>(p.stack().back())->set_slice(slice);
			return true;
		}

		using pattern2 = PatternMatch<NamedExpressionPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("named_expression");
			Subscript::SliceType slice = Subscript::Index{ p.pop_back() };
			auto end_token = p.lexer().peek_token(p.token_position() - 1);
			p.push_to_stack(std::make_shared<Subscript>(
				SourceLocation{ start_token->start(), end_token->end() }));
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
		DEBUG_LOG("SlicesPattern");

		BlockScope scope{ p };

		auto start_token = p.lexer().peek_token(p.token_position());

		using pattern1 = PatternMatch<SlicePattern,
			NegativeLookAhead<SingleTokenPattern<Token::TokenType::COMMA>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("slice !','");
			scope.parent().push_back(p.pop_back());
			return true;
		}

		using pattern2 = PatternMatch<
			ApplyInBetweenPattern<SlicePattern, SingleTokenPattern<Token::TokenType::COMMA>>,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("','.slice+ [',']");
			std::vector<std::variant<Subscript::Index, Subscript::Slice>> dims;
			while (!p.stack().empty()) {
				auto node = p.pop_front();
				ASSERT(as<Subscript>(node))
				auto slice = as<Subscript>(node)->slice();
				if (std::holds_alternative<Subscript::Index>(slice)) {
					dims.push_back(std::get<Subscript::Index>(slice));
				} else if (std::holds_alternative<Subscript::Slice>(slice)) {
					dims.push_back(std::get<Subscript::Slice>(slice));
				} else {
					PARSER_ERROR()
				}
			}
			auto end_token = p.lexer().peek_token(p.token_position() - 1);
			auto subscript = std::make_shared<Subscript>(
				SourceLocation{ start_token->start(), end_token->end() });
			subscript->set_slice(Subscript::ExtSlice{ .dims = dims });
			scope.parent().push_back(subscript);
			return true;
		}

		return false;
	}
};

struct PrimaryPattern_ : Pattern<PrimaryPattern_>
{
	// primary'
	//		| '.' NAME primary'
	// 		| genexp primary'
	// 		| '(' [arguments] ')' primary'
	// 		| '[' slices ']' primary'
	// 		| ϵ
	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("PrimaryPattern_");
		auto start_token = p.lexer().peek_token(p.token_position());
		DEBUG_LOG("{}", start_token->to_string());
		BlockScope primary_scope{ p };
		// '.' NAME primary'
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::DOT>,
			SingleTokenPattern<Token::TokenType::NAME>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'.' NAME");
			auto value = primary_scope.parent().back();
			primary_scope.parent().pop_back();
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			p.push_to_stack(std::make_shared<Attribute>(value,
				name,
				ContextType::LOAD,
				SourceLocation{ start_token->start(), token->end() }));
			using pattern1a = PatternMatch<PrimaryPattern_>;
			if (pattern1a::match(p)) {
				DEBUG_LOG("'.' NAME primary'");
				while (!p.stack().empty()) { primary_scope.parent().push_back(p.pop_front()); }
				return true;
			}
			return false;
		}

		// '(' [arguments] ')' primary'
		using pattern3 = PatternMatch<SingleTokenPattern<Token::TokenType::LPAREN>,
			ZeroOrOnePattern<ArgumentsPattern>,
			SingleTokenPattern<Token::TokenType::RPAREN>>;
		if (pattern3::match(p)) {
			DEBUG_LOG("'(' [arguments] ')'=");
			PRINT_STACK();
			DEBUG_LOG("--------------");
			std::vector<std::shared_ptr<ASTNode>> args;
			std::vector<std::shared_ptr<Keyword>> kwargs;
			while (!p.stack().empty()) {
				auto node = p.pop_front();
				if (auto keyword_node = as<Keyword>(node)) {
					kwargs.push_back(keyword_node);
				} else {
					args.push_back(node);
				}
			}
			auto function = primary_scope.parent().back();
			primary_scope.parent().pop_back();
			auto end_token = p.lexer().peek_token(p.token_position() - 1);
			p.push_to_stack(std::make_shared<Call>(
				function, args, kwargs, SourceLocation{ start_token->start(), end_token->end() }));
			PRINT_STACK();
			using pattern3a = PatternMatch<PrimaryPattern_>;
			if (pattern3a::match(p)) {
				DEBUG_LOG("'(' [arguments] ')' primary'");
				while (!p.stack().empty()) { primary_scope.parent().push_back(p.pop_front()); }
				return true;
			}
			return false;
		}

		// '[' slices ']' primary'
		using pattern4 = PatternMatch<SingleTokenPattern<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPattern<Token::TokenType::RSQB>>;
		if (pattern4::match(p)) {
			DEBUG_LOG("'[' slices ']'");
			auto subscript = p.pop_back();
			auto value = primary_scope.parent().back();
			primary_scope.parent().pop_back();
			ASSERT(as<Subscript>(subscript))
			as<Subscript>(subscript)->set_value(value);
			as<Subscript>(subscript)->set_context(ContextType::LOAD);
			p.push_to_stack(subscript);
			using pattern4a = PatternMatch<PrimaryPattern_>;
			if (pattern4a::match(p)) {
				DEBUG_LOG("'[' slices ']' primary'");
				while (!p.stack().empty()) { primary_scope.parent().push_back(p.pop_front()); }
				return true;
			}
			return false;
		}

		// ϵ
		using pattern5 =
			PatternMatch<LookAhead<OrPattern<SingleTokenPattern<Token::TokenType::NEWLINE,
												 Token::TokenType::DOUBLESTAR,
												 Token::TokenType::STAR,
												 Token::TokenType::AT,
												 Token::TokenType::SLASH,
												 Token::TokenType::DOUBLESLASH,
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
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, FromKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, AndKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, IsKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, IfKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ElseKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, InKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, OrKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, NotKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, AsKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ReturnPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, GlobalKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, PassKeywordPattern>>>>;
		if (pattern5::match(p)) {
			DEBUG_LOG("ϵ");
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
		DEBUG_LOG("PrimaryPattern");
		using pattern2 = PatternMatch<AtomPattern, PrimaryPattern_>;
		if (pattern2::match(p)) {
			DEBUG_LOG("atom primary'")
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
		DEBUG_LOG("AwaitPrimaryPattern");
		// primary
		using pattern2 = PatternMatch<PrimaryPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("primary")
			return true;
		}

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
		DEBUG_LOG("PowerPattern");
		// await_primary '**' factor
		using pattern1 = PatternMatch<AwaitPrimaryPattern,
			SingleTokenPattern<Token::TokenType::DOUBLESTAR>,
			FactorPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("await_primary '**' factor")
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::EXP,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(binary_op);
			return true;
		}

		// await_primary
		using pattern2 = PatternMatch<AwaitPrimaryPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("await_primary")
			return true;
		}

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
		DEBUG_LOG("FactorPattern");
		auto start_token = p.lexer().peek_token(p.token_position());
		// '+' factor
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::PLUS>, FactorPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'+' factor");
			const auto &arg = p.pop_back();
			p.push_to_stack(std::make_shared<UnaryExpr>(UnaryOpType::ADD,
				arg,
				SourceLocation{ start_token->start(), arg->source_location().end }));
			return true;
		}

		// '-' factor
		using pattern2 = PatternMatch<SingleTokenPattern<Token::TokenType::MINUS>, FactorPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("'-' factor");
			const auto &arg = p.pop_back();
			p.push_to_stack(std::make_shared<UnaryExpr>(UnaryOpType::SUB,
				arg,
				SourceLocation{ start_token->start(), arg->source_location().end }));
			return true;
		}

		// '~' factor
		using pattern3 = PatternMatch<SingleTokenPattern<Token::TokenType::TILDE>, FactorPattern>;
		if (pattern3::match(p)) {
			DEBUG_LOG("'~' factor");
			const auto &arg = p.pop_back();
			p.push_to_stack(std::make_shared<UnaryExpr>(UnaryOpType::INVERT,
				arg,
				SourceLocation{ start_token->start(), arg->source_location().end }));
			return true;
		}

		// power
		using pattern4 = PatternMatch<PowerPattern>;
		if (pattern4::match(p)) {
			DEBUG_LOG("power");
			return true;
		}

		return false;
	}
};


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
		DEBUG_LOG("TermPattern_");

		BlockScope scope{ p };
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::STAR>, FactorPattern>;
		if (pattern1::match(p)) {
			auto lhs = scope.parent().back();
			auto rhs = p.pop_front();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::MULTIPLY,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(binary_op);
			if (TermPattern_::matches(p)) {
				scope.parent().pop_back();
				scope.parent().push_back(p.pop_back());
				return true;
			}
		}
		using pattern2 = PatternMatch<SingleTokenPattern<Token::TokenType::SLASH>, FactorPattern>;
		if (pattern2::match(p)) {
			auto lhs = scope.parent().back();
			auto rhs = p.pop_front();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::SLASH,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(binary_op);
			if (TermPattern_::matches(p)) {
				scope.parent().pop_back();
				scope.parent().push_back(p.pop_back());
				return true;
			}
		}

		using pattern3 =
			PatternMatch<SingleTokenPattern<Token::TokenType::DOUBLESLASH>, FactorPattern>;
		if (pattern3::match(p)) {
			auto lhs = scope.parent().back();
			auto rhs = p.pop_front();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::FLOORDIV,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(binary_op);
			if (TermPattern_::matches(p)) {
				scope.parent().pop_back();
				scope.parent().push_back(p.pop_back());
				return true;
			}
		}
		using pattern4 = PatternMatch<SingleTokenPattern<Token::TokenType::PERCENT>, FactorPattern>;
		if (pattern4::match(p)) {
			auto lhs = scope.parent().back();
			auto rhs = p.pop_front();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::MODULO,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(binary_op);
			if (TermPattern_::matches(p)) {
				scope.parent().pop_back();
				scope.parent().push_back(p.pop_back());
				return true;
			}
		}
		using pattern5 = PatternMatch<SingleTokenPattern<Token::TokenType::AT>, FactorPattern>;
		if (pattern5::match(p)) {
			auto lhs = scope.parent().back();
			auto rhs = p.pop_front();
			TODO();
			// auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::AT, lhs, rhs);
			// p.push_to_stack(binary_op);
			if (TermPattern_::matches(p)) {
				// scope.parent().push_back(binary_op);
				return true;
			}
		}

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
		DEBUG_LOG("TermPattern");
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
		DEBUG_LOG("SumPattern_");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		BlockScope scope{ p };
		// '+' term sum'
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::PLUS>, TermPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'+' term sum'");
			PRINT_STACK();
			auto lhs = scope.parent().back();
			auto rhs = p.pop_front();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::PLUS,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(binary_op);
			if (SumPattern_::matches(p)) {
				PRINT_STACK();
				scope.parent().pop_back();
				scope.parent().push_back(p.pop_back());
				return true;
			}
		}

		// '-' term sum'
		using pattern2 = PatternMatch<SingleTokenPattern<Token::TokenType::MINUS>, TermPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("'-' term sum'");
			PRINT_STACK();
			auto lhs = scope.parent().back();
			auto rhs = p.pop_front();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::MINUS,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(binary_op);
			if (SumPattern_::matches(p)) {
				PRINT_STACK();
				scope.parent().pop_back();
				scope.parent().push_back(p.pop_back());
				return true;
			}
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
			DEBUG_LOG("ϵ");
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
		DEBUG_LOG("SumPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		// term sum'
		using pattern1 = PatternMatch<TermPattern, SumPattern_>;
		if (pattern1::match(p)) {
			DEBUG_LOG("term sum'");
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
		DEBUG_LOG("ShiftExprPattern_");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		// '<<' sum shift_expr'
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::LEFTSHIFT>,
			SumPattern,
			ShiftExprPattern_>;
		if (pattern1::match(p)) {
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::LEFTSHIFT,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(binary_op);
			return true;
		}

		// '>>' sum shift_expr'
		using pattern2 = PatternMatch<SingleTokenPattern<Token::TokenType::RIGHTSHIFT>,
			SumPattern,
			ShiftExprPattern_>;
		if (pattern2::match(p)) {
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto binary_op = std::make_shared<BinaryExpr>(BinaryOpType::RIGHTSHIFT,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(binary_op);
			return true;
		}

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
			DEBUG_LOG("shift_expr: ϵ");
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
		DEBUG_LOG("ShiftExprPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		using pattern1 = PatternMatch<SumPattern, ShiftExprPattern_>;
		if (pattern1::match(p)) {
			DEBUG_LOG("shift_expr '<<' sum");
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
		DEBUG_LOG("BitwiseAndPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		// shift_expr
		using pattern2 = PatternMatch<ShiftExprPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("shift_expr");
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
		DEBUG_LOG("BitwiseXorPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		// bitwise_and
		using pattern2 = PatternMatch<BitwiseAndPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("bitwise_and");
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
		DEBUG_LOG("BitwiseOrPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		// bitwise_xor
		using pattern2 = PatternMatch<BitwiseXorPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("bitwise_xor");
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
		DEBUG_LOG("EqBitwiseOrPattern");
		if (pattern1::match(p)) {
			DEBUG_LOG("'==' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs,
				Compare::OpType::Eq,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
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
		DEBUG_LOG("NotEqBitwiseOrPattern");
		if (pattern1::match(p)) {
			DEBUG_LOG("'!=' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs,
				Compare::OpType::NotEq,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
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
		DEBUG_LOG("LtEqBitwiseOrPattern");
		if (pattern1::match(p)) {
			DEBUG_LOG("'<=' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs,
				Compare::OpType::LtE,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
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
		DEBUG_LOG("LtBitwiseOrPattern");
		if (pattern1::match(p)) {
			DEBUG_LOG("'<' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs,
				Compare::OpType::Lt,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
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
		DEBUG_LOG("GtEqBitwiseOrPattern");
		if (pattern1::match(p)) {
			DEBUG_LOG("'>=' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs,
				Compare::OpType::GtE,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
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
		DEBUG_LOG("GtBitwiseOrPattern");
		if (pattern1::match(p)) {
			DEBUG_LOG("'>' bitwise_or");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs,
				Compare::OpType::Gt,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(comparisson);
			return true;
		}
		return false;
	}
};

struct InBitwiseOrPattern : Pattern<InBitwiseOrPattern>
{
	// in_bitwise_or: 'in' bitwise_or
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, InKeywordPattern>,
				BitwiseOrPattern>;
		DEBUG_LOG("InBitwiseOrPattern");
		if (pattern1::match(p)) {
			DEBUG_LOG("'in' bitwise_or ");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs,
				Compare::OpType::In,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(comparisson);
			return true;
		}
		return false;
	}
};

struct NotInBitwiseOrPattern : Pattern<NotInBitwiseOrPattern>
{
	// notin_bitwise_or: 'not' 'in' bitwise_or
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, NotKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, InKeywordPattern>,
				BitwiseOrPattern>;
		DEBUG_LOG("NotInBitwiseOrPattern");
		if (pattern1::match(p)) {
			DEBUG_LOG("'not' 'in' bitwise_or ");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs,
				Compare::OpType::NotIn,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(comparisson);
			return true;
		}
		return false;
	}
};

struct IsNotBitwiseOrPattern : Pattern<IsNotBitwiseOrPattern>
{
	// is_bitwise_or: 'is' 'not' bitwise_or
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, IsKeywordPattern>,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, NotKeywordPattern>,
				BitwiseOrPattern>;
		DEBUG_LOG("IsNotBitwiseOrPattern");
		if (pattern1::match(p)) {
			DEBUG_LOG("'is not' bitwise_or ");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs,
				Compare::OpType::IsNot,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
			p.push_to_stack(comparisson);
			return true;
		}
		return false;
	}
};

struct IsBitwiseOrPattern : Pattern<IsBitwiseOrPattern>
{
	// is_bitwise_or: 'is' bitwise_or
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, IsKeywordPattern>,
				BitwiseOrPattern>;
		DEBUG_LOG("IsBitwiseOrPattern");
		if (pattern1::match(p)) {
			DEBUG_LOG("'is' bitwise_or ");
			auto rhs = p.pop_back();
			auto lhs = p.pop_back();
			auto comparisson = std::make_shared<Compare>(lhs,
				Compare::OpType::Is,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
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
		DEBUG_LOG("CompareOpBitwiseOrPairPattern");
		// eq_bitwise_or
		using pattern1 = PatternMatch<EqBitwiseOrPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("eq_bitwise_or");
			return true;
		}
		// noteq_bitwise_or
		using pattern2 = PatternMatch<NotEqBitwiseOrPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("not_eq_bitwise_or");
			return true;
		}
		// lte_bitwise_or
		using pattern3 = PatternMatch<LtEqBitwiseOrPattern>;
		if (pattern3::match(p)) {
			DEBUG_LOG("lte_bitwise_or");
			return true;
		}
		// lt_bitwise_or
		using pattern4 = PatternMatch<LtBitwiseOrPattern>;
		if (pattern4::match(p)) {
			DEBUG_LOG("lt_bitwise_or");
			return true;
		}
		// gte_bitwise_or
		using pattern5 = PatternMatch<GtEqBitwiseOrPattern>;
		if (pattern5::match(p)) {
			DEBUG_LOG("gte_bitwise_or");
			return true;
		}
		// gt_bitwise_or
		using pattern6 = PatternMatch<GtBitwiseOrPattern>;
		if (pattern6::match(p)) {
			DEBUG_LOG("gt_bitwise_or");
			return true;
		}
		// notin_bitwise_or
		using pattern7 = PatternMatch<NotInBitwiseOrPattern>;
		if (pattern7::match(p)) {
			DEBUG_LOG("notin_bitwise_or");
			return true;
		}
		// in_bitwise_or`
		using pattern8 = PatternMatch<InBitwiseOrPattern>;
		if (pattern8::match(p)) {
			DEBUG_LOG("in_bitwise_or");
			return true;
		}
		// isnot_bitwise_or
		using pattern9 = PatternMatch<IsNotBitwiseOrPattern>;
		if (pattern9::match(p)) {
			DEBUG_LOG("isnot_bitwise_or");
			return true;
		}
		// is_bitwise_or
		using pattern10 = PatternMatch<IsBitwiseOrPattern>;
		if (pattern10::match(p)) {
			DEBUG_LOG("is_bitwise_or");
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
		DEBUG_LOG("ComparissonPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		// bitwise_or compare_op_bitwise_or_pair+
		using pattern1 =
			PatternMatch<BitwiseOrPattern, OneOrMorePattern<CompareOpBitwiseOrPairPattern>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("bitwise_or compare_op_bitwise_or_pair+");
			return true;
		}
		// bitwise_or
		using pattern2 = PatternMatch<BitwiseOrPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("bitwise_or");
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
		DEBUG_LOG("InversionPattern");
		auto start_token = p.lexer().peek_token(p.token_position());
		DEBUG_LOG("{}", start_token->to_string());
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, NotKeywordPattern>,
				ComparissonPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'not' inversion")
			const auto &node = p.pop_back();
			p.push_to_stack(std::make_shared<UnaryExpr>(UnaryOpType::NOT,
				node,
				SourceLocation{ start_token->start(), node->source_location().end }));
			return true;
		}
		// comparison
		using pattern2 = PatternMatch<ComparissonPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("comparison")
			return true;
		}

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
		BlockScope scope{ p };
		DEBUG_LOG("ConjunctionPattern");
		const auto start_token = p.lexer().peek_token(p.token_position());
		DEBUG_LOG("{}", start_token->to_string());
		// inversion ('and' inversion )+
		using pattern1 = PatternMatch<InversionPattern,
			OneOrMorePattern<
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, AndKeywordPattern>,
				InversionPattern>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("inversion ('and' inversion )+")
			std::vector<std::shared_ptr<ASTNode>> values;
			values.reserve(p.stack().size());
			while (!p.stack().empty()) { values.push_back(p.pop_front()); }
			const auto end_token = p.lexer().peek_token(p.token_position());
			scope.parent().push_back(std::make_shared<BoolOp>(BoolOp::OpType::And,
				values,
				SourceLocation{ start_token->start(), end_token->end() }));
			return true;
		}

		// inversion
		using pattern2 = PatternMatch<InversionPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("inversion")
			while (!p.stack().empty()) { scope.parent().push_back(p.pop_front()); }
			return true;
		}

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
		BlockScope scope{ p };
		DEBUG_LOG("DisjunctionPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		// conjunction ('or' conjunction )+
		using pattern1 = PatternMatch<ConjunctionPattern,
			OneOrMorePattern<
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, OrKeywordPattern>,
				ConjunctionPattern>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("conjunction ('or' conjunction )+")
			std::vector<std::shared_ptr<ASTNode>> values;
			values.reserve(p.stack().size());
			while (!p.stack().empty()) { values.push_back(p.pop_front()); }
			scope.parent().push_back(std::make_shared<BoolOp>(BoolOp::OpType::Or,
				values,
				SourceLocation{ values.front()->source_location().start,
					values.front()->source_location().end }));
			return true;
		}

		// conjunction
		using pattern2 = PatternMatch<ConjunctionPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("conjunction")
			while (!p.stack().empty()) { scope.parent().push_back(p.pop_front()); }
			return true;
		}

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
		DEBUG_LOG("ExpressionPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		using pattern1 = PatternMatch<DisjunctionPattern,
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, IfKeywordPattern>,
			DisjunctionPattern,
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ElseKeywordPattern>,
			ExpressionPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("disjunction 'if' disjunction 'else' expression")
			auto orelse = p.pop_back();
			auto test = p.pop_back();
			auto body = p.pop_back();
			p.push_to_stack(std::make_shared<IfExpr>(test,
				body,
				orelse,
				SourceLocation{ test->source_location().start, orelse->source_location().end }));
			return true;
		}

		// disjunction
		using pattern2 = PatternMatch<DisjunctionPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("disjunction")
			return true;
		}

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
		if (pattern2::match(p)) {
			DEBUG_LOG("expression")
			return true;
		}
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
		BlockScope scope{ p };
		auto start_token = p.lexer().peek_token(p.token_position());
		// star_expression (',' star_expression )+ [',']
		using pattern1 = PatternMatch<StarExpressionPattern,
			OneOrMorePattern<SingleTokenPattern<Token::TokenType::COMMA>, StarExpressionPattern>,
			ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("star_expression (',' star_expression )+ [',']");
			std::vector<std::shared_ptr<ASTNode>> expressions;
			while (!p.stack().empty()) { expressions.push_back(p.pop_front()); }
			auto end_token = p.lexer().peek_token(p.token_position() - 1);
			auto result = std::make_shared<Tuple>(expressions,
				ContextType::LOAD,
				SourceLocation{ start_token->start(), end_token->end() });
			scope.parent().push_back(result);
			return true;
		}
		// star_expression ','
		using pattern2 =
			PatternMatch<StarExpressionPattern, SingleTokenPattern<Token::TokenType::COMMA>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("star_expression ','");
			while (!p.stack().empty()) { scope.parent().push_back(p.pop_back()); }
			return true;
		}
		// star_expression
		using pattern3 = PatternMatch<StarExpressionPattern>;
		if (pattern3::match(p)) {
			DEBUG_LOG("star_expression");
			while (!p.stack().empty()) { scope.parent().push_back(p.pop_back()); }
			return true;
		}
		return false;
	}
};

struct SingleSubscriptAttributeTargetPattern : Pattern<SingleSubscriptAttributeTargetPattern>
{
	// single_subscript_attribute_target:
	//     | t_primary '.' NAME !t_lookahead
	//     | t_primary '[' slices ']' !t_lookahead
	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("SingleSubscriptAttributeTargetPattern");

		// t_primary '.' NAME !t_lookahead
		using pattern1 = PatternMatch<TPrimaryPattern,
			SingleTokenPattern<Token::TokenType::DOT>,
			SingleTokenPattern<Token::TokenType::NAME>,
			NegativeLookAhead<TLookahead>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("t_primary '.' NAME !t_lookahead");
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			const auto primary = p.pop_back();
			DEBUG_LOG("{}", name);
			primary->print_node("");
			auto attribute = std::make_shared<Attribute>(primary,
				name,
				ContextType::STORE,
				SourceLocation{ primary->source_location().start, token->end() });
			p.push_to_stack(attribute);
			return true;
		}

		// t_primary '[' slices ']' !t_lookahead
		using pattern2 = PatternMatch<TPrimaryPattern,
			SingleTokenPattern<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPattern<Token::TokenType::RSQB>,
			NegativeLookAhead<TLookahead>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("t_primary '[' slices ']' !t_lookahead");
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
		DEBUG_LOG("SingleTargetPattern");
		using pattern1 = PatternMatch<SingleSubscriptAttributeTargetPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("single_subscript_attribute_target");
			return true;
		}

		using pattern2 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("NAME");
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			p.push_to_stack(std::make_shared<Name>(
				name, ContextType::STORE, SourceLocation{ token->start(), token->end() }));
			return true;
		}

		using pattern3 = PatternMatch<SingleTokenPattern<Token::TokenType::LPAREN>,
			SingleTargetPattern,
			SingleTokenPattern<Token::TokenType::RPAREN>>;
		if (pattern3::match(p)) {
			DEBUG_LOG("'(' single_target ')'");
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
		DEBUG_LOG("AugAssignPattern")

		// '+='
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::PLUSEQUAL>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'+='");
			const auto &lhs = p.pop_back();
			// defer rhs assignment to caller. Am I shooting myself in the foot?
			// at least a null dereference goes with a bang...
			p.push_to_stack(std::make_shared<AugAssign>(lhs,
				BinaryOpType::PLUS,
				nullptr,
				SourceLocation{ lhs->source_location().start, lhs->source_location().end }));
			return true;
		}

		// '-='
		using pattern2 = PatternMatch<SingleTokenPattern<Token::TokenType::MINEQUAL>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("'-='");
			const auto &lhs = p.pop_back();
			// defer rhs assignment to caller. Am I shooting myself in the foot?
			// at least a null dereference goes with a bang...
			p.push_to_stack(std::make_shared<AugAssign>(lhs,
				BinaryOpType::MINUS,
				nullptr,
				SourceLocation{ lhs->source_location().start, lhs->source_location().end }));
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
		auto start_token = p.lexer().peek_token(p.token_position());
		using EqualMatch = SingleTokenPattern<Token::TokenType::EQUAL>;
		using pattern3 =
			PatternMatch<OneOrMorePattern<StarTargetsPattern, EqualMatch>, StarExpressionsPattern>;
		size_t start_position = p.stack().size();
		if (pattern3::match(p)) {
			DEBUG_LOG("(star_targets '=' )+ (yield_expr | star_expressions) !'=' [TYPE_COMMENT]");
			std::vector<std::shared_ptr<ASTNode>> target_elements;
			auto expressions = p.pop_back();
			const auto &stack = p.stack();
			for (size_t i = start_position; i < stack.size(); ++i) {
				target_elements.push_back(stack[i]);
			}
			auto end_token = p.lexer().peek_token(p.token_position());

			auto targets = std::make_shared<Tuple>(target_elements,
				ContextType::STORE,
				SourceLocation{ target_elements.front()->source_location().start,
					target_elements.back()->source_location().end });

			while (p.stack().size() > start_position) { p.pop_back(); }
			PRINT_STACK();
			expressions->print_node("");

			auto assignment = [&]() {
				if (targets->elements().size() == 1) {
					return std::make_shared<Assign>(
						std::vector<std::shared_ptr<ASTNode>>{ targets->elements().back() },
						expressions,
						"",
						SourceLocation{ start_token->start(), end_token->end() });
				} else {
					return std::make_shared<Assign>(
						std::vector<std::shared_ptr<ASTNode>>{ targets },
						expressions,
						"",
						SourceLocation{ start_token->start(), end_token->end() });
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
			DEBUG_LOG("single_target augassign ~ (yield_expr | star_expressions)");
			const auto &rhs = p.pop_back();
			auto aug_assign = p.pop_back();
			as<AugAssign>(aug_assign)->set_value(rhs);
			p.push_to_stack(aug_assign);
			return true;
		}
		return false;
	}
};

struct ReturnStatementPattern : Pattern<ReturnStatementPattern>
{
	// return_stmt:
	// 		| 'return' [star_expressions]
	static bool matches_impl(Parser &p)
	{
		auto start_token = p.lexer().peek_token(p.token_position());
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ReturnPattern>,
				ZeroOrMorePattern<StarExpressionsPattern>>;
		if (pattern1::match(p)) {
			const auto &return_value = p.pop_back();
			auto return_node = std::make_shared<Return>(return_value,
				SourceLocation{ start_token->start(), return_value->source_location().end });
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
		DEBUG_LOG("dotted_name");
		const auto token = p.lexer().peek_token(p.token_position() + 1);
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::DOT>,
			SingleTokenPattern<Token::TokenType::NAME>,
			DottedNamePattern_>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'.' NAME dotted_name'");
			std::string name{ p.lexer().get(token->start(), token->end()) };
			p.push_to_stack(
				std::make_shared<Constant>(name, SourceLocation{ token->start(), token->end() }));
			return true;
		}
		using pattern2 = PatternMatch<
			LookAhead<SingleTokenPattern<Token::TokenType::NEWLINE, Token::TokenType::NAME>>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("ϵ");
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
		DEBUG_LOG("dotted_name");
		const auto token = p.lexer().peek_token(p.token_position());
		std::string name{ p.lexer().get(token->start(), token->end()) };
		size_t stack_position = p.stack().size();
		using pattern2 =
			PatternMatch<SingleTokenPattern<Token::TokenType::NAME>, DottedNamePattern_>;
		if (pattern2::match(p)) {
			DEBUG_LOG("NAME dotted_name'");
			std::static_pointer_cast<Import>(p.stack()[stack_position - 1])->add_dotted_name(name);
			while (p.stack().size() > stack_position) {
				const auto &node = p.pop_back();
				std::string value =
					std::get<String>(*static_pointer_cast<Constant>(node)->value()).s;
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
		DEBUG_LOG("dotted_as_name");
		using pattern1 = PatternMatch<DottedNamePattern>;
		if (pattern1::match(p)) {
			using pattern1a = PatternMatch<
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, AsKeywordPattern>,
				SingleTokenPattern<Token::TokenType::NAME>>;
			if (pattern1a::match(p)) {
				DEBUG_LOG("['as' NAME ]");
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
		DEBUG_LOG("dotted_as_names");
		using pattern1 = PatternMatch<OneOrMorePattern<ApplyInBetweenPattern<DottedAsNamePattern,
			SingleTokenPattern<Token::TokenType::COMMA>>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("','.dotted_as_name+");
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
		BlockScope import_scope{ p };
		auto start_token = p.lexer().peek_token(p.token_position());
		p.push_to_stack(
			std::make_shared<Import>(SourceLocation{ start_token->start(), start_token->end() }));
		DEBUG_LOG("import_name");
		using pattern1 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ImportKeywordPattern>,
			DottedAsNamesPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'import' dotted_as_names");
			import_scope.parent().push_back(p.pop_back());
			ASSERT(p.stack().empty())
			return true;
		}
		return false;
	}
};


struct ImportFromKeywordPattern : Pattern<ImportFromKeywordPattern>
{
	// import_from:
	// | 'from' ('.' | '...')* dotted_name 'import' import_from_targets
	// | 'from' ('.' | '...')+ 'import' import_from_targets
	static bool matches_impl(Parser &)
	{
		DEBUG_LOG("import_from");
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
		DEBUG_LOG("import_stmt");
		using pattern1 = PatternMatch<ImportNamePattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("import_name");
			return true;
		}
		using pattern2 = PatternMatch<ImportFromKeywordPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("import_from");
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
		DEBUG_LOG("raise_stmt");
		const auto initial_stack_size = p.stack().size();
		const auto start_token = p.lexer().peek_token(p.token_position());
		using pattern1 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, RaiseKeywordPattern>,
			ExpressionPattern,
			ZeroOrOnePattern<
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, FromKeywordPattern>,
				ExpressionPattern>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'raise' expression ['from' expression ] ");
			ASSERT((p.stack().size() - initial_stack_size) > 0)
			ASSERT((p.stack().size() - initial_stack_size) < 3)
			if ((p.stack().size() - initial_stack_size) == 1) {
				const auto &exception = p.pop_back();
				p.push_to_stack(std::make_shared<Raise>(exception,
					nullptr,
					SourceLocation{ start_token->start(), exception->source_location().end }));

			} else {
				const auto cause = p.pop_back();
				const auto &exception = p.pop_back();
				p.push_to_stack(std::make_shared<Raise>(exception,
					cause,
					SourceLocation{ start_token->start(), cause->source_location().end }));
			}
			return true;
		}

		using pattern2 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, RaiseKeywordPattern>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("'raise'");
			ASSERT(p.stack().size() == initial_stack_size)
			p.push_to_stack(std::make_shared<Raise>(
				SourceLocation{ start_token->start(), start_token->end() }));
			return true;
		}
		return false;
	}
};

struct DeleteAtomPattern : Pattern<DeleteAtomPattern>
{
	// del_t_atom:
	//     | NAME
	//     | '(' del_target ')'
	//     | '(' [del_targets] ')'
	//     | '[' [del_targets] ']'
	static bool matches_impl(Parser &p)
	{
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>>;
		if (pattern1::match(p)) {
			auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string name_str{ token->start().pointer_to_program,
				token->end().pointer_to_program };
			p.push_to_stack(std::make_shared<Name>(
				name_str, ContextType::DELETE, SourceLocation{ token->start(), token->end() }));
			return true;
		}
		return false;
	}
};

struct DeleteTargetsPattern : Pattern<DeleteTargetsPattern>
{
	// del_target:
	//     | t_primary '.' NAME !t_lookahead
	//     | t_primary '[' slices ']' !t_lookahead
	//     | del_t_atom
	static bool matches_impl(Parser &p)
	{
		BlockScope scope{ p };
		const auto start_token = p.lexer().peek_token(p.token_position());
		using pattern1 = PatternMatch<TPrimaryPattern,
			SingleTokenPattern<Token::TokenType::DOT>,
			SingleTokenPattern<Token::TokenType::NAME>,
			NegativeLookAhead<TLookahead>>;
		if (pattern1::match(p)) {
			TODO();
			return true;
		}

		using pattern2 = PatternMatch<TPrimaryPattern,
			SingleTokenPattern<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPattern<Token::TokenType::RSQB>,
			NegativeLookAhead<TLookahead>>;
		if (pattern2::match(p)) {
			auto slices = p.pop_back();
			auto value = p.pop_back();
			const auto end_token = p.lexer().peek_token(p.token_position());
			ASSERT(as<Subscript>(slices))
			as<Subscript>(slices)->set_value(value);
			scope.parent().push_back(
				std::make_shared<Delete>(std::vector<std::shared_ptr<ASTNode>>{ slices },
					SourceLocation{ start_token->start(), end_token->end() }));
			return true;
		}

		using pattern3 = PatternMatch<DeleteAtomPattern>;
		if (pattern3::match(p)) {
			std::vector<std::shared_ptr<ASTNode>> to_delete;
			to_delete.reserve(p.stack().size());
			while (!p.stack().empty()) { to_delete.push_back(p.pop_front()); }
			const auto end_token = p.lexer().peek_token(p.token_position());
			scope.parent().push_back(std::make_shared<Delete>(
				to_delete, SourceLocation{ start_token->start(), end_token->end() }));
			return true;
		}
		return false;
	}
};

struct DeleteStatementPattern : Pattern<DeleteStatementPattern>
{
	// del_stmt:
	// 	| 'del' del_targets &(';' | NEWLINE)
	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("'del_stmt'");
		using pattern1 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, DeleteKeywordPattern>,
			DeleteTargetsPattern,
			LookAhead<OrPattern<SingleTokenPattern<Token::TokenType::SEMI>,
				SingleTokenPattern<Token::TokenType::NEWLINE>>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'del' del_targets &(';' | NEWLINE)");
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
		DEBUG_LOG("AssertStatementPattern");
		const auto initial_stack_size = p.stack().size();
		const auto start_token = p.lexer().peek_token(p.token_position());
		using pattern1 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, AssertKeywordPattern>,
			ExpressionPattern,
			ZeroOrMorePattern<SingleTokenPattern<Token::TokenType::COMMA>, ExpressionPattern>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("assert_stmt: 'assert' expression [',' expression ]");
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
			const auto end_token = p.lexer().peek_token(p.token_position());
			p.push_to_stack(std::make_shared<Assert>(
				test, msg, SourceLocation{ start_token->start(), end_token->end() }));
			return true;
		}
		return false;
	}
};

struct GlobalStatementPattern : Pattern<GlobalStatementPattern>
{
	struct GlobalName : Pattern<GlobalName>
	{
		static bool matches_impl(Parser &p)
		{
			auto token = p.lexer().peek_token(p.token_position());
			std::string_view maybe_name{ token->start().pointer_to_program,
				token->end().pointer_to_program };
			// global_name: NAME
			using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::NAME>>;

			if (pattern1::match(p)) {
				ASSERT(as<Global>(p.stack().back()))
				as<Global>(p.stack().back())->add_name(std::string(maybe_name));
				return true;
			}
			return false;
		}
	};

	// global_stmt: 'global' ','.NAME+
	static bool matches_impl(Parser &p)
	{
		BlockScope scope{ p };
		const auto start_token = p.lexer().peek_token(p.token_position());
		p.push_to_stack(std::make_shared<Global>(std::vector<std::string>{},
			SourceLocation{ start_token->start(), start_token->end() }));

		DEBUG_LOG("GlobalStatementPattern");
		using pattern1 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, GlobalKeywordPattern>,
			OneOrMorePattern<
				ApplyInBetweenPattern<GlobalName, SingleTokenPattern<Token::TokenType::COMMA>>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("global_stmt: 'global' ','.NAME+");
			scope.parent().push_back(p.pop_back());
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
			DEBUG_LOG("assignment");
			return true;
		}
		using pattern2 = PatternMatch<StarExpressionsPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("star_expressions");
			return true;
		}
		using pattern3 = PatternMatch<ReturnStatementPattern>;
		if (pattern3::match(p)) {
			DEBUG_LOG("return_stmt");
			return true;
		}
		using pattern4 = PatternMatch<ImportStatementPattern>;
		if (pattern4::match(p)) {
			DEBUG_LOG("import_stmt");
			return true;
		}
		using pattern5 = PatternMatch<RaiseStatementPattern>;
		if (pattern5::match(p)) {
			DEBUG_LOG("raise_stmt");
			return true;
		}
		using pattern6 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, PassKeywordPattern>>;
		if (pattern6::match(p)) {
			DEBUG_LOG("pass");
			const auto start_token = p.lexer().peek_token(p.token_position());
			p.push_to_stack(
				std::make_shared<Pass>(SourceLocation{ start_token->start(), start_token->end() }));
			return true;
		}
		using pattern7 = PatternMatch<DeleteStatementPattern>;
		if (pattern7::match(p)) {
			DEBUG_LOG("del_stmt");
			return true;
		}
		using pattern9 = PatternMatch<AssertStatementPattern>;
		if (pattern9::match(p)) {
			DEBUG_LOG("assert_stmt");
			return true;
		}
		using pattern10 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, BreakKeywordPattern>>;
		if (pattern10::match(p)) {
			DEBUG_LOG("break");
			spdlog::error("'break' not implemented");
			TODO();
			return true;
		}
		using pattern11 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ContinueKeywordPattern>>;
		if (pattern11::match(p)) {
			DEBUG_LOG("continue");
			const auto start_token = p.lexer().peek_token(p.token_position());
			p.push_to_stack(std::make_shared<Continue>(
				SourceLocation{ start_token->start(), start_token->end() }));
			return true;
		}
		using pattern12 = PatternMatch<GlobalStatementPattern>;
		if (pattern12::match(p)) {
			DEBUG_LOG("global_stmt");
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

struct AnnotationPattern : Pattern<AnnotationPattern>
{
	// annotation: ':' expression
	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("annotation");
		BlockScope scope{ p };
		// ':' expression
		using pattern1 =
			PatternMatch<SingleTokenPattern<Token::TokenType::COLON>, ExpressionPattern>;
		if (pattern1::match(p)) {
			ASSERT(p.stack().size() == 1)
			scope.parent().push_back(p.pop_back());
			return true;
		}
		return false;
	}
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
			using pattern1a = PatternMatch<AnnotationPattern>;
			std::shared_ptr<ASTNode> annotation = [&p]() -> std::shared_ptr<ASTNode> {
				if (pattern1a::match(p)) {
					const auto &type = p.pop_back();
					return type;
				} else {
					return nullptr;
				}
			}();
			p.push_to_stack(std::make_shared<Argument>(
				argname, annotation, "", SourceLocation{ token->start(), token->end() }));
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
			DEBUG_LOG("param ',' TYPE_COMMENT?");
			ASSERT(as<Argument>(p.stack().back()));
			return true;
		}

		// param TYPE_COMMENT? &')'
		// TODO: implement TYPE_COMMENT?
		using pattern2 =
			PatternMatch<ParamPattern, LookAhead<SingleTokenPattern<Token::TokenType::RPAREN>>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("param TYPE_COMMENT? &')'");
			ASSERT(as<Argument>(p.stack().back()));
			return true;
		}
		return false;
	}
};


struct DefaultPattern : Pattern<DefaultPattern>
{
	// default: '=' expression
	static bool matches_impl(Parser &p)
	{
		BlockScope scope{ p };
		DEBUG_LOG("default")
		using pattern1 =
			PatternMatch<SingleTokenPattern<Token::TokenType::EQUAL>, ExpressionPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'=' expression");
			ASSERT(p.stack().size() == 1)
			scope.parent().push_back(p.pop_back());
			return true;
		}
		return false;
	}
};

struct ParamWithDefaultPattern : Pattern<ParamWithDefaultPattern>
{
	// param_with_default:
	//     | param default ',' TYPE_COMMENT?
	//     | param default TYPE_COMMENT? &')'
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<ParamPattern, DefaultPattern, SingleTokenPattern<Token::TokenType::COMMA>>;
		// param default ',' TYPE_COMMENT?
		if (pattern1::match(p)) {
			DEBUG_LOG("param default ',' TYPE_COMMENT?");
			auto default_ = p.pop_back();
			auto arg = p.pop_back();
			ASSERT(as<Argument>(arg));
			p.push_to_stack(std::make_shared<Keyword>(as<Argument>(arg)->name(),
				default_,
				SourceLocation{ arg->source_location().start, default_->source_location().end }));
			return true;
		}

		using pattern2 = PatternMatch<ParamPattern,
			DefaultPattern,
			LookAhead<SingleTokenPattern<Token::TokenType::RPAREN>>>;
		// param default TYPE_COMMENT? &')'
		if (pattern2::match(p)) {
			DEBUG_LOG("param default TYPE_COMMENT? &')'");
			auto default_ = p.pop_back();
			auto arg = p.pop_back();
			ASSERT(as<Argument>(arg));
			p.push_to_stack(std::make_shared<Keyword>(as<Argument>(arg)->name(),
				default_,
				SourceLocation{ arg->source_location().start, default_->source_location().end }));
			return true;
		}
		return false;
	}
};

struct ParamMaybeDefaultPattern : Pattern<ParamMaybeDefaultPattern>
{
	// param_maybe_default:
	//     | param default? ',' TYPE_COMMENT?
	//     | param default? TYPE_COMMENT? &')'
	static bool matches_impl(Parser &p)
	{
		BlockScope scope{ p };
		using pattern1 = PatternMatch<ParamPattern,
			ZeroOrOnePattern<DefaultPattern>,
			SingleTokenPattern<Token::TokenType::COMMA>>;
		// param default? ',' TYPE_COMMENT?
		if (pattern1::match(p)) {
			DEBUG_LOG("param default? ',' TYPE_COMMENT?");
			const bool has_default = p.stack().size() == 2;
			auto default_ = [&]() -> std::shared_ptr<ASTNode> {
				if (has_default) {
					return p.pop_back();
				} else {
					return nullptr;
				}
			}();
			auto arg = p.pop_back();
			ASSERT(as<Argument>(arg));
			if (has_default) {
				scope.parent().push_back(std::make_shared<Keyword>(as<Argument>(arg)->name(),
					default_,
					SourceLocation{
						arg->source_location().start, default_->source_location().end }));
			} else {
				scope.parent().push_back(arg);
			}
			return true;
		}

		using pattern2 = PatternMatch<ParamPattern,
			ZeroOrOnePattern<DefaultPattern>,
			LookAhead<SingleTokenPattern<Token::TokenType::RPAREN>>>;
		// param default? TYPE_COMMENT? &')'
		if (pattern2::match(p)) {
			DEBUG_LOG("param default? TYPE_COMMENT? &')'");
			const bool has_default = p.stack().size() == 2;
			auto default_ = [&]() -> std::shared_ptr<ASTNode> {
				if (has_default) {
					return p.pop_back();
				} else {
					return nullptr;
				}
			}();
			auto arg = p.pop_back();
			ASSERT(as<Argument>(arg));
			if (has_default) {
				scope.parent().push_back(std::make_shared<Keyword>(as<Argument>(arg)->name(),
					default_,
					SourceLocation{
						arg->source_location().start, default_->source_location().end }));
			} else {
				scope.parent().push_back(arg);
			}
			return true;
		}
		return false;
	}
};

struct KeywordsPattern : Pattern<KeywordsPattern>
{
	// kwds: '**' param_no_default
	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("KeywordsPattern")
		// '**' param_no_default
		using pattern1 =
			PatternMatch<SingleTokenPattern<Token::TokenType::DOUBLESTAR>, ParamNoDefaultPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'**' param_no_default")
			return true;
		}
		return false;
	}
};

struct StarEtcPattern : Pattern<StarEtcPattern>
{
	// star_etc:
	//     | '*' param_no_default param_maybe_default* [kwds]
	//     | '*' ',' param_maybe_default+ [kwds]
	//     | kwds
	static bool matches_impl(Parser &p)
	{
		BlockScope scope{ p };
		DEBUG_LOG("StarEtcPattern")
		// '*' param_no_default param_maybe_default* [kwds]
		using pattern1 = PatternMatch<SingleTokenPattern<Token::TokenType::STAR>,
			ParamNoDefaultPattern,
			ZeroOrMorePattern<ParamMaybeDefaultPattern>>;
		if (pattern1::match(p)) {

			DEBUG_LOG("'*' param_no_default param_maybe_default*")
			auto &args = scope.parent().back();
			ASSERT(as<Arguments>(args))
			auto param_no_default = p.pop_front();
			ASSERT(as<Argument>(param_no_default))
			as<Arguments>(args)->set_arg(as<Argument>(param_no_default));
			while (!p.stack().empty()) {
				auto node = p.pop_front();
				if (as<Argument>(node)) {
					as<Arguments>(args)->push_kwonlyarg(as<Argument>(node));
					as<Arguments>(args)->push_kwarg_default(nullptr);
				} else if (as<Keyword>(node)) {
					auto arg = std::make_shared<Argument>(*as<Keyword>(node)->arg(),
						nullptr,
						"",
						SourceLocation{
							node->source_location().start, node->source_location().end });
					as<Arguments>(args)->push_kwonlyarg(arg);
					as<Arguments>(args)->push_kwarg_default(as<Keyword>(node)->value());
				} else {
					PARSER_ERROR();
				}
			}
			using pattern1a = PatternMatch<KeywordsPattern>;
			if (pattern1a::match(p)) {
				DEBUG_LOG("'*' param_no_default param_maybe_default* [kwds]")
				auto arg = p.pop_back();
				ASSERT(as<Argument>(arg));
				as<Arguments>(args)->set_kwarg(as<Argument>(arg));
			}
			return true;
		}

		// '*' ',' param_maybe_default+ [kwds]
		using pattern2 = PatternMatch<SingleTokenPattern<Token::TokenType::STAR>,
			SingleTokenPattern<Token::TokenType::COMMA>,
			ZeroOrMorePattern<ParamMaybeDefaultPattern>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("'*' ',' param_maybe_default+")
			auto &args = scope.parent().back();
			ASSERT(as<Arguments>(args))
			while (!p.stack().empty()) {
				auto node = p.pop_front();
				if (as<Argument>(node)) {
					as<Arguments>(args)->push_kwonlyarg(as<Argument>(node));
					as<Arguments>(args)->push_kwarg_default(nullptr);
				} else if (as<Keyword>(node)) {
					auto arg = std::make_shared<Argument>(*as<Keyword>(node)->arg(),
						nullptr,
						"",
						SourceLocation{
							node->source_location().start, node->source_location().end });
					as<Arguments>(args)->push_kwonlyarg(arg);
					as<Arguments>(args)->push_kwarg_default(as<Keyword>(node)->value());
				} else {
					PARSER_ERROR();
				}
			}
			using pattern2a = PatternMatch<KeywordsPattern>;
			if (pattern2a::match(p)) {
				DEBUG_LOG("'*' ',' param_maybe_default+ [kwds]")
				auto arg = p.pop_back();
				ASSERT(as<Argument>(arg));
				as<Arguments>(args)->set_kwarg(as<Argument>(arg));
			}
			return true;
		}

		// kwds
		using pattern3 = PatternMatch<KeywordsPattern>;
		if (pattern3::match(p)) {
			DEBUG_LOG("kwds")
			auto &args = scope.parent().back();
			ASSERT(as<Arguments>(args))
			auto arg = p.pop_back();
			ASSERT(as<Argument>(arg));
			as<Arguments>(args)->set_kwarg(as<Argument>(arg));
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
		const auto start_token = p.lexer().peek_token(p.token_position());
		p.push_to_stack(std::make_shared<Arguments>(
			SourceLocation{ start_token->start(), start_token->end() }));
		size_t stack_size = p.stack().size();
		auto &args = p.stack().back();

		// param_no_default+ param_with_default* [star_etc]
		using pattern3 = PatternMatch<OneOrMorePattern<ParamNoDefaultPattern>,
			ZeroOrMorePattern<ParamWithDefaultPattern>>;
		if (pattern3::match(p)) {
			DEBUG_LOG("param_no_default+ param_with_default*");
			PRINT_STACK();
			for (size_t idx = stack_size; idx < p.stack().size(); ++idx) {
				auto node = p.stack()[idx];
				if (as<Argument>(node)) {
					as<Arguments>(args)->push_arg(as<Argument>(node));
				} else if (as<Keyword>(node)) {
					auto arg = std::make_shared<Argument>(
						*as<Keyword>(node)->arg(), nullptr, "", node->source_location());
					as<Arguments>(args)->push_arg(arg);
					as<Arguments>(args)->push_default(as<Keyword>(node)->value());
				} else {
					PARSER_ERROR();
				}
			}
			while (p.stack().size() > stack_size) { p.pop_back(); }
			using pattern3a = PatternMatch<StarEtcPattern>;
			if (pattern3a::match(p)) {
				DEBUG_LOG("param_no_default+ param_with_default* [star_etc]");
			}
			PRINT_STACK();
			return true;
		}

		// star_etc
		using pattern5 = PatternMatch<StarEtcPattern>;
		if (pattern5::match(p)) {
			DEBUG_LOG("star_etc");
			PRINT_STACK();
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
			DEBUG_LOG("parameters");
			PRINT_STACK();
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
			DEBUG_LOG("NEWLINE INDENT statements DEDENT");
			for (const auto &node : p.stack()) { node->print_node(""); }
			return true;
		}

		using pattern2 = PatternMatch<SimpleStatementPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("simple_stmt");
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
			DEBUG_LOG("function_name: NAME");
			const auto token = p.lexer().peek_token(p.token_position() - 1);
			std::string function_name{ token->start().pointer_to_program,
				token->end().pointer_to_program };
			p.push_to_stack(std::make_shared<Constant>(
				function_name, SourceLocation{ token->start(), token->end() }));
			return true;
		}
		return false;
	}
};

struct FunctionDefinitionPattern : Pattern<FunctionDefinitionPattern>
{
	// function_def: 'def' function_name '(' [params] ')' ['->' expression ] ':'
	// [func_type_comment]
	static bool matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, DefPattern>,
				FunctionNamePattern,
				SingleTokenPattern<Token::TokenType::LPAREN>,
				ZeroOrMorePattern<ParamsPattern>,
				SingleTokenPattern<Token::TokenType::RPAREN>,
				ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::RARROW>, ExpressionPattern>,
				SingleTokenPattern<Token::TokenType::COLON>>;
		if (pattern1::match(p)) {
			DEBUG_LOG(
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
	//     | ASYNC 'def' function_name '(' [params] ')' ['->' expression ] ':'
	//     [func_type_comment] block
	static bool matches_impl(Parser &p)
	{
		BlockScope scope{ p };
		// function_def block
		using pattern1 = PatternMatch<FunctionDefinitionPattern>;
		const auto &start = p.lexer().peek_token(0)->start();
		if (pattern1::match(p)) {
			DEBUG_LOG("function_def_raw: function_def");
			auto name = p.pop_front();
			auto args = [&]() -> std::shared_ptr<ast::ASTNode> {
				if (!p.stack().empty()) {
					return p.pop_front();
				} else {
					return std::make_shared<Arguments>(SourceLocation{ name->source_location() });
				}
			}();
			auto returns = [&]() -> std::shared_ptr<ast::ASTNode> {
				if (!p.stack().empty()) {
					return p.pop_front();
				} else {
					return nullptr;
				}
			}();

			if (args) { args->print_node(""); }
			name->print_node("");
			std::vector<std::shared_ptr<ASTNode>> body;
			{
				BlockScope inner_scope{ p };
				using pattern1a = PatternMatch<ZeroOrOnePattern<BlockPattern>>;
				if (pattern1a::match(p)) { DEBUG_LOG("block"); }
				for (auto &&node : p.stack()) { body.push_back(std::move(node)); }
			}

			const auto &end = p.lexer().peek_token(0)->end();
			ASSERT(as<Constant>(name));
			ASSERT(as<Arguments>(args));
			auto function = std::make_shared<FunctionDefinition>(
				std::get<String>(*as<Constant>(name)->value()).s,
				as<Arguments>(args),
				body,
				std::vector<std::shared_ptr<ASTNode>>{},
				returns,
				"",
				SourceLocation{ start, end });
			scope.parent().push_back(function);
			function->print_node("");
			return true;
		}
		return false;
	}
};

struct DecoratorsPattern : Pattern<DecoratorsPattern>
{
	// decorators: ('@' named_expression NEWLINE )+
	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("DecoratorsPattern")

		// ('@' named_expression NEWLINE )+
		using pattern1 = PatternMatch<OneOrMorePattern<SingleTokenPattern<Token::TokenType::AT>,
			NamedExpressionPattern,
			SingleTokenPattern<Token::TokenType::NEWLINE>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("('@' named_expression NEWLINE )+")
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
		BlockScope scope{ p };
		// decorators function_def_raw
		using pattern1 = PatternMatch<DecoratorsPattern, FunctionDefinitionRawStatement>;
		if (pattern1::match(p)) {
			DEBUG_LOG("decorators function_def_raw");
			ASSERT(p.stack().size() > 1)
			auto function = p.pop_back();
			ASSERT(as<FunctionDefinition>(function))
			while (!p.stack().empty()) {
				as<FunctionDefinition>(function)->add_decorator(p.pop_front());
			}
			scope.parent().push_back(function);
			return true;
		}

		// function_def_raw
		using pattern2 = PatternMatch<FunctionDefinitionRawStatement>;
		if (pattern2::match(p)) {
			DEBUG_LOG("function_def_raw");
			ASSERT(p.stack().size() == 1)
			scope.parent().push_back(p.pop_back());
			return true;
		}
		return false;
	}
};

struct ElseBlockStatementPattern : Pattern<ElseBlockStatementPattern>
{
	// else_block: 'else' ':' block
	static bool matches_impl(Parser &p)
	{
		// else_block: 'else' ':' block
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ElseKeywordPattern>,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("else_block: 'else' ':' block");
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
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ElifKeywordPattern>,
				NamedExpressionPattern,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern0::match(p)) {
			DEBUG_LOG("'if' named_expression ':' block");
			std::vector<std::shared_ptr<ASTNode>> orelse;
			std::vector<std::shared_ptr<ASTNode>> body;
			{
				BlockScope inner_scope{ p };
				using pattern1 = PatternMatch<ElifStatementPattern>;
				if (pattern1::match(p)) { DEBUG_LOG("elif_stmt"); }
				using pattern2 = PatternMatch<ZeroOrOnePattern<ElseBlockStatementPattern>>;
				if (pattern2::match(p)) { DEBUG_LOG("[else_block]"); }
				for (auto &&node : p.stack()) { orelse.push_back(std::move(node)); }
			}
			auto test = p.pop_front();
			while (!p.stack().empty()) { body.push_back(p.pop_front()); }
			ASSERT(!body.empty());
			SourceLocation location = [&]() {
				if (orelse.empty()) {
					return SourceLocation{ test->source_location().start,
						body.back()->source_location().end };
				} else {
					return SourceLocation{ test->source_location().start,
						orelse.back()->source_location().end };
				}
			}();
			scope.parent().push_back(std::make_shared<If>(test, body, orelse, location));
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
		DEBUG_LOG("IfStatementPattern");
		BlockScope scope{ p };
		const auto start_token = p.lexer().peek_token(p.token_position());
		// 'if' named_expression ':' block
		using pattern0 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, IfKeywordPattern>,
				NamedExpressionPattern,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern0::match(p)) {
			DEBUG_LOG("'if' named_expression ':' block");
			std::vector<std::shared_ptr<ASTNode>> orelse;
			std::vector<std::shared_ptr<ASTNode>> body;
			{
				BlockScope inner_scope{ p };
				using pattern1 = PatternMatch<ElifStatementPattern>;
				if (pattern1::match(p)) { DEBUG_LOG("elif_stmt"); }
				using pattern2 = PatternMatch<ZeroOrOnePattern<ElseBlockStatementPattern>>;
				if (pattern2::match(p)) { DEBUG_LOG("[else_block]"); }
				for (auto &&node : p.stack()) { orelse.push_back(std::move(node)); }
			}
			auto test = p.pop_front();
			while (!p.stack().empty()) { body.push_back(p.pop_front()); }
			const auto end_token = p.lexer().peek_token(p.token_position());
			scope.parent().push_back(std::make_shared<If>(
				test, body, orelse, SourceLocation{ start_token->start(), end_token->end() }));
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
		DEBUG_LOG("ClassDefinitionRawPattern");
		BlockScope scope{ p };
		ASSERT(p.stack().empty())
		// 'class' NAME ['(' [arguments] ')' ] ':'

		if (!p.lexer().peek_token(p.token_position() + 1)) { return false; }

		auto maybe_name_token = p.lexer().peek_token(p.token_position() + 1);
		std::string class_name{ maybe_name_token->start().pointer_to_program,
			maybe_name_token->end().pointer_to_program };
		using pattern0 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ClassPattern>,
				SingleTokenPattern<Token::TokenType::NAME>,
				ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::LPAREN>,
					ZeroOrMorePattern<ArgumentsPattern>,
					SingleTokenPattern<Token::TokenType::RPAREN>>,
				SingleTokenPattern<Token::TokenType::COLON>>;
		if (pattern0::match(p)) {
			DEBUG_LOG("'class' NAME ['(' [arguments] ')' ] ':'");
			// FIXME: assumes no inheritance
			DEBUG_LOG("class name: {}", class_name);

			std::vector<std::shared_ptr<ASTNode>> bases;
			std::vector<std::shared_ptr<Keyword>> keywords;

			while (!p.stack().empty()) {
				const auto &node = p.pop_front();
				if (auto keyword_node = as<Keyword>(node)) {
					keywords.push_back(keyword_node);
				} else {
					bases.push_back(node);
				}
			}

			std::vector<std::shared_ptr<ASTNode>> body;
			{
				BlockScope block_scope{ p };
				using pattern1 = PatternMatch<BlockPattern>;
				if (pattern1::match(p)) {
					DEBUG_LOG("block");
				} else {
					return false;
				}
				for (auto &&node : p.stack()) { body.push_back(std::move(node)); }
			}
			std::vector<std::shared_ptr<ASTNode>> decorator_list;
			// while (!p.stack().empty()) { arguments.push_back(p.pop_front()); }
			scope.parent().push_back(std::make_shared<ClassDefinition>(class_name,
				bases,
				keywords,
				body,
				decorator_list,
				SourceLocation{ maybe_name_token->start(), maybe_name_token->end() }));
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
		BlockScope scope{ p };
		// decorators class_def_raw
		using pattern1 = PatternMatch<DecoratorsPattern, ClassDefinitionRawPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("decorators class_def_raw");
			ASSERT(p.stack().size() > 1)
			auto class_definition = p.pop_back();
			ASSERT(as<ClassDefinition>(class_definition))
			while (!p.stack().empty()) {
				as<ClassDefinition>(class_definition)->add_decorator(p.pop_front());
			}
			scope.parent().push_back(class_definition);
			return true;
		}

		// class_def_raw
		using pattern2 = PatternMatch<ClassDefinitionRawPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("class_def_raw");
			ASSERT(p.stack().size() == 1)
			scope.parent().push_back(p.pop_back());
			return true;
		}
		return false;
	}
};

struct ForStatementPattern : Pattern<ForStatementPattern>
{
	// for_stmt:
	//     | 'for' star_targets 'in' ~ star_expressions ':' [TYPE_COMMENT] block [else_block]
	//     | ASYNC 'for' star_targets 'in' ~ star_expressions ':' [TYPE_COMMENT] block
	//     [else_block]
	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("ForStatementPattern");
		BlockScope scope{ p };
		const auto start_token = p.lexer().peek_token(p.token_position());
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ForKeywordPattern>,
				StarTargetsPattern,
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, InKeywordPattern>,
				StarExpressionsPattern,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'for' star_targets 'in' ~ star_expressions ':' [TYPE_COMMENT] block");
			std::vector<std::shared_ptr<ASTNode>> orelse;
			{
				BlockScope else_scope{ p };
				using pattern1 = PatternMatch<ZeroOrOnePattern<ElseBlockStatementPattern>>;
				if (pattern1::match(p)) { DEBUG_LOG("[else_block]"); }
				for (auto &&node : p.stack()) { orelse.push_back(std::move(node)); }
			}
			// FIXME: type comment is currently not considered
			std::string type_comment{ "" };
			auto target = p.pop_front();
			auto iter = p.pop_front();
			std::vector<std::shared_ptr<ASTNode>> body;
			while (!p.stack().empty()) { body.push_back(p.pop_front()); }
			const auto end_token = p.lexer().peek_token(p.token_position());
			scope.parent().push_back(std::make_shared<For>(target,
				iter,
				body,
				orelse,
				type_comment,
				SourceLocation{ start_token->start(), end_token->end() }));
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
		DEBUG_LOG("ExceptBlockPattern");
		const auto start_token = p.lexer().peek_token(p.token_position());
		{
			BlockScope scope{ p };
			using pattern1 = PatternMatch<
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ExceptKeywordPattern>,
				ExpressionPattern>;
			if (pattern1::match(p)) {
				DEBUG_LOG("'except' expression");
				std::string name{};
				using pattern1a = PatternMatch<
					AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, AsKeywordPattern>,
					SingleTokenPattern<Token::TokenType::NAME>>;
				if (pattern1a::match(p)) {
					DEBUG_LOG("['as' NAME ]");
					auto token = p.lexer().peek_token(p.token_position() - 1);
					DEBUG_LOG("{}", token->to_string());
					name = std::string{ token->start().pointer_to_program,
						token->end().pointer_to_program };
				}
				using pattern1b =
					PatternMatch<SingleTokenPattern<Token::TokenType::COLON>, BlockPattern>;
				if (pattern1b::match(p)) {
					DEBUG_LOG("':' block");
					std::vector<std::shared_ptr<ASTNode>> body;
					const auto type = p.pop_front();
					while (!p.stack().empty()) { body.push_back(p.pop_front()); }
					const auto end_token = p.lexer().peek_token(p.token_position());
					scope.parent().push_front(std::make_shared<ExceptHandler>(type,
						name,
						body,
						SourceLocation{ start_token->start(), end_token->end() }));
					return true;
				}
			}
		}
		{
			BlockScope scope{ p };
			using pattern2 = PatternMatch<
				AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, ExceptKeywordPattern>,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
			if (pattern2::match(p)) {
				DEBUG_LOG("'except' ':' block");
				std::vector<std::shared_ptr<ASTNode>> body;
				while (!p.stack().empty()) { body.push_back(p.pop_front()); }
				const auto end_token = p.lexer().peek_token(p.token_position());
				scope.parent().push_front(std::make_shared<ExceptHandler>(
					nullptr, "", body, SourceLocation{ start_token->start(), end_token->end() }));
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
		DEBUG_LOG("FinallyBlockPattern");
		using pattern1 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, FinallyKeywordPattern>,
			SingleTokenPattern<Token::TokenType::COLON>,
			BlockPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'finally' ':' block");
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
		DEBUG_LOG("TryStatementPattern");
		BlockScope scope{ p };
		const auto start_token = p.lexer().peek_token(p.token_position());
		// 'try' ':' block
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, TryKeywordPattern>,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'try' ':' block");

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
					DEBUG_LOG("finally_block");
					while (!p.stack().empty()) { finally.push_back(p.pop_front()); }
					match = true;
				}
			}
			// except_block+
			{
				BlockScope except_block{ p };
				using pattern1b = PatternMatch<OneOrMorePattern<ExceptBlockPattern>>;
				if (pattern1b::match(p)) {
					DEBUG_LOG("except_block");
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
							DEBUG_LOG("[else_block]");
							while (!p.stack().empty()) { orelse.push_back(p.pop_front()); }
						}
					}
					// [finally_block]
					{
						BlockScope finally_block{ p };
						using pattern1d = PatternMatch<FinallyBlockPattern>;
						if (pattern1d::match(p)) {
							DEBUG_LOG("[finally_block]");
							while (!p.stack().empty()) { finally.push_back(p.pop_front()); }
						}
					}
				}
			}
			if (!match) { return false; }
			while (!p.stack().empty()) { body.push_back(p.pop_front()); }
			const auto end_token = p.lexer().peek_token(p.token_position());
			scope.parent().push_back(std::make_shared<Try>(body,
				handlers,
				orelse,
				finally,
				SourceLocation{ start_token->start(), end_token->end() }));
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
		DEBUG_LOG("WhileStatementPattern");
		BlockScope scope{ p };
		const auto start_token = p.lexer().peek_token(p.token_position());

		// 'while' named_expression ':' block
		using pattern0 = PatternMatch<
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, WhileKeywordPattern>,
			NamedExpressionPattern,
			SingleTokenPattern<Token::TokenType::COLON>,
			BlockPattern>;
		if (pattern0::match(p)) {
			DEBUG_LOG("'while' named_expression ':' block");
			std::vector<std::shared_ptr<ASTNode>> orelse;
			std::vector<std::shared_ptr<ASTNode>> body;
			{
				BlockScope inner_scope{ p };
				using pattern1 = PatternMatch<ZeroOrOnePattern<ElseBlockStatementPattern>>;
				if (pattern1::match(p)) { DEBUG_LOG("[else_block]"); }
				for (auto &&node : p.stack()) { orelse.push_back(std::move(node)); }
			}
			auto test = p.pop_front();
			while (!p.stack().empty()) { body.push_back(p.pop_front()); }
			const auto end_token = p.lexer().peek_token(p.token_position());
			scope.parent().push_back(std::make_shared<While>(
				test, body, orelse, SourceLocation{ start_token->start(), end_token->end() }));
			return true;
		}

		return false;
	}
};

struct WithItemPattern : Pattern<WithItemPattern>
{
	// with_item:
	//     | expression 'as' star_target &(',' | ')' | ':')
	//     | expression
	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("WithItemPattern")
		const auto start_token = p.lexer().peek_token(p.token_position());
		using pattern1 = PatternMatch<ExpressionPattern,
			AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, AsKeywordPattern>,
			StarTargetPattern,
			LookAhead<OrPattern<SingleTokenPattern<Token::TokenType::COMMA>,
				SingleTokenPattern<Token::TokenType::RPAREN>,
				SingleTokenPattern<Token::TokenType::COLON>>>>;
		if (pattern1::match(p)) {
			DEBUG_LOG("expression 'as' star_target &(',' | ')' | ':')")
			auto var = p.pop_back();
			auto context_expr = p.pop_back();
			const auto end_token = p.lexer().peek_token(p.token_position());
			p.push_to_stack(std::make_shared<WithItem>(
				context_expr, var, SourceLocation{ start_token->start(), end_token->end() }));
			return true;
		}

		using pattern2 = PatternMatch<ExpressionPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("expression")
			const auto end_token = p.lexer().peek_token(p.token_position());
			p.push_to_stack(std::make_shared<WithItem>(
				p.pop_back(), nullptr, SourceLocation{ start_token->start(), end_token->end() }));
			return true;
		}

		return false;
	}
};

struct WithStatementPattern : Pattern<WithStatementPattern>
{
	// with_stmt:
	//     | 'with' '(' ','.with_item+ ','? ')' ':' block
	//     | 'with' ','.with_item+ ':' [TYPE_COMMENT] block
	//     | ASYNC 'with' '(' ','.with_item+ ','? ')' ':' block
	//     | ASYNC 'with' ','.with_item+ ':' [TYPE_COMMENT] block
	static bool matches_impl(Parser &p)
	{
		DEBUG_LOG("WithStatementPattern")

		BlockScope with_scope{ p };
		const auto start_token = p.lexer().peek_token(p.token_position());

		// 'with' '(' ','.with_item+ ','? ')' ':' block
		using pattern1 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, WithKeywordPattern>,
				SingleTokenPattern<Token::TokenType::LPAREN>,
				OneOrMorePattern<ApplyInBetweenPattern<WithItemPattern,
					SingleTokenPattern<Token::TokenType::COMMA>>>,
				ZeroOrOnePattern<SingleTokenPattern<Token::TokenType::COMMA>>,
				SingleTokenPattern<Token::TokenType::RPAREN>,
				SingleTokenPattern<Token::TokenType::COLON>,
				BlockPattern>;
		if (pattern1::match(p)) {
			DEBUG_LOG("'with' '(' ','.with_item+ ','? ')' ':' block")
			TODO();
			return true;
		}

		// 'with' ','.with_item+ ':' [TYPE_COMMENT] block
		using pattern2 =
			PatternMatch<AndLiteral<SingleTokenPattern<Token::TokenType::NAME>, WithKeywordPattern>,
				OneOrMorePattern<ApplyInBetweenPattern<WithItemPattern,
					SingleTokenPattern<Token::TokenType::COMMA>>>,
				SingleTokenPattern<Token::TokenType::COLON>>;
		if (pattern2::match(p)) {
			DEBUG_LOG("'with' ','.with_item+ ':' [TYPE_COMMENT]")
			std::vector<std::shared_ptr<WithItem>> with_items;
			with_items.reserve(p.stack().size());
			while (!p.stack().empty()) {
				auto node = p.pop_front();
				ASSERT(as<WithItem>(node))
				with_items.push_back(as<WithItem>(node));
			}

			BlockScope block_scope{ p };

			using pattern2a = PatternMatch<BlockPattern>;
			if (pattern2a::match(p)) {
				DEBUG_LOG("block")

				std::vector<std::shared_ptr<ASTNode>> body;
				body.reserve(p.stack().size());
				while (!p.stack().empty()) { body.push_back(p.pop_front()); }

				const auto end_token = p.lexer().peek_token(p.token_position());
				with_scope.parent().push_back(std::make_shared<With>(with_items,
					body,
					"",
					SourceLocation{ start_token->start(), end_token->end() }));
				return true;
			}
			return false;
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
			DEBUG_LOG("function_def");
			PRINT_STACK();
			return true;
		}
		// if_stmt
		using pattern2 = PatternMatch<IfStatementPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("if_stmt");
			PRINT_STACK();
			return true;
		}

		// class_def
		using pattern3 = PatternMatch<ClassDefinitionPattern>;
		if (pattern3::match(p)) {
			DEBUG_LOG("class_def");
			PRINT_STACK();
			return true;
		}

		// class_def
		using pattern4 = PatternMatch<WithStatementPattern>;
		if (pattern4::match(p)) {
			DEBUG_LOG("with_stmt");
			PRINT_STACK();
			return true;
		}

		// for_stmt
		using pattern5 = PatternMatch<ForStatementPattern>;
		if (pattern5::match(p)) {
			DEBUG_LOG("for_stmt");
			PRINT_STACK();
			return true;
		}

		// try_stmt
		using pattern6 = PatternMatch<TryStatementPattern>;
		if (pattern6::match(p)) {
			DEBUG_LOG("try_stmt");
			PRINT_STACK();
			return true;
		}

		// while_stmt
		using pattern7 = PatternMatch<WhileStatementPattern>;
		if (pattern7::match(p)) {
			DEBUG_LOG("while_stmt");
			PRINT_STACK();
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
			DEBUG_LOG("compound_stmt");
			return true;
		}
		using pattern2 = PatternMatch<SimpleStatementPattern>;
		if (pattern2::match(p)) {
			DEBUG_LOG("simple_stmt");
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
		size_t idx = 0;
		auto t = *p.lexer().peek_token(idx);
		const size_t row = t.start().row;
		std::string line = "";
		while (row == t.start().row) {
			line += std::string(t.start().pointer_to_program, t.end().pointer_to_program);
			idx++;
			t = *p.lexer().peek_token(idx);
		}
		spdlog::error("Syntax error on line {}: '{}'", row + 1, line);
		PARSER_ERROR();
	}
};

namespace parser {
void Parser::parse()
{
	const auto result = FilePattern::matches(*this);
	(void)result;
	DEBUG_LOG("Parser return code: {}", result);
	m_module->print_node("");
}
}// namespace parser
