#include "Parser.hpp"
#include "runtime/SyntaxError.hpp"
#include "runtime/Value.hpp"

#include <gmpxx.h>

#include <sstream>

using namespace py;
using namespace ast;
using namespace parser;

// #undef NDEBUG
#define PARSER_ERROR()                                           \
	do {                                                         \
		spdlog::error("Parser error {}:{}", __FILE__, __LINE__); \
		std::abort();                                            \
	} while (0)

#ifndef NDEBUG
#define DEBUG_LOG(...)              \
	do {                            \
		spdlog::trace(__VA_ARGS__); \
	} while (0)
#else
#define DEBUG_LOG(MSG, ...)
#endif

#define TODO_NO_FAIL()                                                                      \
	do {                                                                                    \
		spdlog::error("Parser error: Unimplemented parser rule {}:{}", __FILE__, __LINE__); \
		return {};                                                                          \
	} while (0)

static int hits = 0;

size_t Parser::CacheHash::operator()(const Parser::CacheKey &cache) const
{
	size_t seed = cache.rule.hash_code();
	seed ^= bit_cast<size_t>(cache.token.start().pointer_to_program) + 0x9e3779b9 + (seed << 6)
			+ (seed >> 2);
	seed ^= static_cast<size_t>(cache.token.token_type()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	return seed;
}

bool Parser::CacheEqual::operator()(const Parser::CacheKey &lhs, const Parser::CacheKey &rhs) const
{
	if ((lhs.token.start().pointer_to_program != rhs.token.start().pointer_to_program)
		|| (lhs.token.token_type() != rhs.token.token_type())) {
		return false;
	}
	return lhs.rule == rhs.rule;
}

// template<typename Derived> struct Pattern
// {
// 	virtual ~Pattern() = default;

// 	static bool matches(Parser &p)
// 	{
// 		const auto start_stack_size = p.stack().size();
// 		const auto start_position = p.token_position();
// 		const bool is_match = Derived::matches_impl(p);
// 		if (!is_match) {
// 			while (p.stack().size() > start_stack_size) { p.pop_back(); }
// 			p.token_position() = start_position;
// 		}
// 		return is_match;
// 	}
// };

template<typename T> struct traits;

namespace detail {
template<typename... _Ts> struct as_tuple;

template<typename... _Ts> struct as_tuple<std::variant<_Ts...>>
{
	using type = std::tuple<_Ts...>;
};

using ValueTypesTuple = typename as_tuple<Parser::CacheValue::ValueType>::type;

template<typename T, typename Tuple> struct has_type;

template<typename T, typename... Us>
struct has_type<T, std::tuple<Us...>> : std::disjunction<std::is_same<T, Us>...>
{
};
}// namespace detail

template<typename Derived> struct PatternV2
{
	virtual ~PatternV2() = default;

	using ResultType = typename traits<Derived>::result_type;

  private:
	// algorithm to detect direct left recursion, based on
	// http://web.cs.ucla.edu/~todd/research/pepm08.pdf
	static std::optional<ResultType> grow_lr(Parser &p, size_t position, Parser::CacheValue &value)
	{
		while (true) {
			p.token_position() = position;
			const std::optional<ResultType> result = Derived::matches_impl(p);
			// detect that no progress was made as a result of evaluating the rule
			if (!result.has_value() || (p.token_position() <= value.position)) { break; }
			value.value = *result;
			value.position = p.token_position();
		}
		p.token_position() = value.position;
		if (std::holds_alternative<Parser::CacheValue::ValueType>(value.value)) {
			auto &v = std::get<Parser::CacheValue::ValueType>(value.value);
			ASSERT(std::holds_alternative<ResultType>(v));
			return std::get<ResultType>(v);
		} else {
			return {};
		}
	}

  public:
	static std::optional<ResultType> matches(Parser &p)
	{
		const auto start_position = p.token_position();
		if constexpr (::detail::has_type<ResultType, ::detail::ValueTypesTuple>{}) {
			const auto token = p.lexer().peek_token(start_position);
			ASSERT(token.has_value());
			Parser::CacheKey line{ typeid(Derived), *token };
			Parser::CacheValue value{ false, start_position };
			p.m_cache[line] = value;
		}
		const std::optional<ResultType> result = Derived::matches_impl(p);
		if constexpr (::detail::has_type<ResultType, ::detail::ValueTypesTuple>{}) {
			const auto token = p.lexer().peek_token(start_position);
			if (result.has_value()) {
				Parser::CacheKey line{ typeid(Derived), *token };
				auto &value = p.m_cache.at(line);
				ASSERT(value.has_value());
				if (std::holds_alternative<bool>(value->value) && std::get<bool>(value->value)) {
					return grow_lr(p, start_position, *value);
				} else {
					return result;
				}
			}
		}

		if (!result.has_value()) {
			p.token_position() = start_position;
			return {};
		}
		return result;
	}
};

template<size_t TypeIdx, typename PatternTuple, typename> class PatternMatchV2_;

template<size_t TypeIdx, typename PatternTuple>
class PatternMatchV2_<TypeIdx,
	PatternTuple,
	std::enable_if_t<TypeIdx == std::tuple_size_v<PatternTuple>>>
{
  public:
	static std::optional<std::tuple<>> match(Parser &) { return {}; }
};


template<size_t TypeIdx, typename PatternTuple, typename = void> class PatternMatchV2_
{
	template<typename T, typename = void> struct has_advance_by : std::false_type
	{
	};

	template<typename T>
	struct has_advance_by<T, decltype(std::declval<T>().advance_by, void())> : std::true_type
	{
	};

	template<typename... input_t>
	using tuple_cat_t = decltype(std::tuple_cat(std::declval<input_t>()...));

	using CurrentType = typename std::tuple_element<TypeIdx, PatternTuple>::type;
	using ResultTypeHead = std::invoke_result_t<decltype(CurrentType::matches), Parser &>;
	using PatternMatchTail_ = PatternMatchV2_<TypeIdx + 1, PatternTuple>;
	using ResultTypeTail = std::invoke_result_t<decltype(PatternMatchTail_::match), Parser &>;
	using ResultType = tuple_cat_t<std::tuple<typename ResultTypeHead::value_type>,
		typename ResultTypeTail::value_type>;

  public:
	PatternMatchV2_() {}

	static std::optional<ResultType> advance(Parser &p,
		const typename ResultTypeHead::value_type &result)
	{
		if constexpr (has_advance_by<CurrentType>::value) {
			p.token_position() += CurrentType::advance_by;
		}
		if constexpr (TypeIdx + 1 == std::tuple_size_v<PatternTuple>) {
			return std::make_optional(std::make_tuple(result));
		} else {
			auto tail = PatternMatchTail_::match(p);
			if (!tail.has_value()) { return std::nullopt; }
			return std::tuple_cat(std::make_tuple(result), *tail);
		}
	}

	static std::optional<ResultType> match(Parser &p)
	{
		const size_t original_token_position = p.token_position();

		const auto t = p.lexer().peek_token(original_token_position);
		if (!t.has_value()) { return {}; }
		std::optional<Parser::CacheKey> line;
		if constexpr (::detail::has_type<typename ResultTypeHead::value_type,
						  ::detail::ValueTypesTuple>{}) {
			line.emplace(Parser::CacheKey{ typeid(CurrentType), *t });
			if (auto it = p.m_cache.find(*line); it != p.m_cache.end()) {
				hits++;
				auto &cache = it->second;
				if (!cache.has_value()) { return {}; }
				// auto&& [node, position] = *cache;
				auto &value = cache->value;
				const auto &position = cache->position;
				p.token_position() = position;
				if (std::holds_alternative<bool>(value)) {
					std::get<bool>(value) = true;
					return {};
				} else {
					auto &v = std::get<Parser::CacheValue::ValueType>(value);
					ASSERT(std::holds_alternative<typename ResultTypeHead::value_type>(v));
					return advance(p, std::get<typename ResultTypeHead::value_type>(v));
				}
			}
		}

		if (auto result = CurrentType::matches(p)) {
			if constexpr (::detail::has_type<typename ResultTypeHead::value_type,
							  ::detail::ValueTypesTuple>{}) {
				p.m_cache[*line] = Parser::CacheValue{ *result, p.token_position() };
			}
			return advance(p, *result);
		} else {
			p.token_position() = original_token_position;
			if constexpr (::detail::has_type<typename ResultTypeHead::value_type,
							  ::detail::ValueTypesTuple>{}) {
				p.m_cache[*line] = std::nullopt;
			}
			return std::nullopt;
		}
	}
};

template<typename... PatternType> class PatternMatchV2
{
	template<typename T> struct TypeExtractor_;

	template<typename... T> struct TypeExtractor_<std::tuple<T...>>
	{
		using type = std::tuple<typename traits<T>::result_type...>;
	};

	using ResultType = typename TypeExtractor_<std::tuple<PatternType...>>::type;

  public:
	PatternMatchV2() {}
	static std::optional<ResultType> match(Parser &p)
	{
		const auto start_token_position = p.token_position();
		auto result = PatternMatchV2_<0, std::tuple<PatternType...>>::match(p);
		if (!result.has_value()) {
			p.token_position() = start_token_position;
			return {};
		}
		return result;
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

template<typename... PatternTypes> struct OneOrMorePatternV2;

template<typename... PatternTypes> struct traits<OneOrMorePatternV2<PatternTypes...>>
{
  private:
	using _result_type =
		typename std::invoke_result_t<decltype(PatternMatchV2<PatternTypes...>::match),
			Parser &>::value_type;

  public:
	using result_type = std::conditional_t<sizeof...(PatternTypes) == 1,
		std::vector<typename std::tuple_element_t<0, _result_type>>,
		std::vector<_result_type>>;
};


template<typename... PatternTypes>
struct OneOrMorePatternV2 : PatternV2<OneOrMorePatternV2<PatternTypes...>>
{
	using ResultType = typename traits<OneOrMorePatternV2<PatternTypes...>>::result_type;

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using PatternType = PatternMatchV2<PatternTypes...>;
		ResultType result_collection;
		if (auto result = PatternType::match(p); !result.has_value()) {
			return {};
		} else {
			if constexpr (sizeof...(PatternTypes) == 1) {
				result_collection.push_back(std::get<0>(*result));
			} else {
				result_collection.push_back(*result);
			}
		}
		auto original_token_position = p.token_position();
		while (true) {
			if (auto result = PatternType::match(p); !result.has_value()) {
				break;
			} else {
				if constexpr (sizeof...(PatternTypes) == 1) {
					result_collection.push_back(std::get<0>(*result));
				} else {
					result_collection.push_back(*result);
				}
			}
			original_token_position = p.token_position();
		}
		p.token_position() = original_token_position;
		return result_collection;
	}
};

template<typename MainKeywordPatternType, typename InBetweenPattern> struct ApplyInBetweenPatternV2;

template<typename MainKeywordPatternType, typename InBetweenPattern>
struct traits<ApplyInBetweenPatternV2<MainKeywordPatternType, InBetweenPattern>>
{
	using result_type = std::vector<typename traits<MainKeywordPatternType>::result_type>;
};

template<typename MainKeywordPatternType, typename InBetweenPattern>
struct ApplyInBetweenPatternV2
	: PatternV2<ApplyInBetweenPatternV2<MainKeywordPatternType, InBetweenPattern>>
{
	using ResultType = typename traits<
		ApplyInBetweenPatternV2<MainKeywordPatternType, InBetweenPattern>>::result_type;

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using MainKeywordPatternType_ = PatternMatchV2<MainKeywordPatternType>;
		using InBetweenPattern_ = PatternMatchV2<InBetweenPattern>;
		ResultType result_collection;
		std::optional<size_t> original_token_position;

		do {
			if (auto result = MainKeywordPatternType_::match(p); !result.has_value()) {
				if (original_token_position.has_value()) {
					p.token_position() = *original_token_position;
				} else {
					// failed to match main pattern at the start, so this is a pattern mismatch
					return {};
				}
				break;
			} else {
				auto [el] = *result;
				result_collection.push_back(el);
			}
			original_token_position = p.token_position();
		} while (InBetweenPattern_::match(p));

		DEBUG_LOG(
			"ApplyInBetweenPattern: {}", p.lexer().peek_token(p.token_position())->to_string());

		return result_collection;
	}
};


template<typename... PatternTypes> struct ZeroOrMorePatternV2;

template<typename... PatternTypes> struct traits<ZeroOrMorePatternV2<PatternTypes...>>
{
  private:
	using _PatternType = PatternMatchV2<PatternTypes...>;
	using _result_type =
		typename std::invoke_result_t<decltype(_PatternType::match), Parser &>::value_type;

  public:
	using result_type = std::conditional_t<sizeof...(PatternTypes) == 1,
		std::vector<typename std::tuple_element_t<0, _result_type>>,
		std::vector<_result_type>>;
};

template<typename... PatternTypes>
struct ZeroOrMorePatternV2 : PatternV2<ZeroOrMorePatternV2<PatternTypes...>>
{
	using ResultType = typename traits<ZeroOrMorePatternV2<PatternTypes...>>::result_type;

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using PatternType = PatternMatchV2<PatternTypes...>;
		ResultType result_vector;
		if (auto result = PatternType::match(p); !result.has_value()) {
			return result_vector;
		} else {
			if constexpr (sizeof...(PatternTypes) == 1) {
				result_vector.push_back(std::get<0>(*result));
			} else {
				result_vector.push_back(*result);
			}
		}
		auto original_token_position = p.token_position();
		while (true) {
			auto result = PatternType::match(p);
			if (!result.has_value()) { break; }
			if constexpr (sizeof...(PatternTypes) == 1) {
				result_vector.push_back(std::get<0>(*result));
			} else {
				result_vector.push_back(*result);
			}
			original_token_position = p.token_position();
		}
		p.token_position() = original_token_position;
		return result_vector;
	}
};

template<typename... PatternTypes> struct ZeroOrOnePatternV2;

template<typename... PatternTypes> struct traits<ZeroOrOnePatternV2<PatternTypes...>>
{
  private:
	using _PatternType = PatternMatchV2<PatternTypes...>;
	using _result_type =
		typename std::invoke_result_t<decltype(_PatternType::match), Parser &>::value_type;

  public:
	using result_type = std::conditional_t<sizeof...(PatternTypes) == 1,
		std::optional<typename std::tuple_element_t<0, _result_type>>,
		std::optional<_result_type>>;
};

template<typename... PatternTypes>
struct ZeroOrOnePatternV2 : PatternV2<ZeroOrOnePatternV2<PatternTypes...>>
{
	using ResultType = typename traits<ZeroOrOnePatternV2<PatternTypes...>>::result_type;
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using PatternType = PatternMatchV2<PatternTypes...>;
		auto original_token_position = p.token_position();
		if (auto result = PatternType::match(p)) {
			if constexpr (sizeof...(PatternTypes) == 1) {
				return std::make_optional(std::get<0>(*result));
			} else {
				return std::make_optional(*result);
			}
		}
		p.token_position() = original_token_position;
		DEBUG_LOG("ZeroOrOnePatternV2 (no match): {}",
			p.lexer().peek_token(p.token_position())->to_string());
		ResultType result{};
		return std::make_optional(result);
	}
};

template<typename T> struct equal_types;
template<typename T> struct ast_nodes;

template<typename T, typename... Args> struct equal_types<std::tuple<T, Args...>>
{
	using type = std::conjunction<std::is_same<T, Args>...>;
	static constexpr bool value = type{};
};

template<typename... Args> struct ast_nodes<std::tuple<Args...>>
{
  private:
	template<typename T> struct is_shared_ptr : std::false_type
	{
	};

	template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type
	{
	};

	template<typename... _Args> struct _nested
	{
		using type = std::conjunction<std::is_base_of<ASTNode, typename _Args::element_type>...>;
	};

	static constexpr bool _all_shared_ptr = std::conjunction_v<is_shared_ptr<Args>...>;

  public:
	static constexpr bool value =
		typename std::conditional_t<equal_types<std::tuple<Args...>>::value,
			std::true_type,
			std::conditional_t<_all_shared_ptr, _nested<Args...>, std::false_type>>::type{};
};

namespace detail {
template<typename T, typename Args> struct is_in
{
	static constexpr bool value = std::is_same_v<T, Args>;
};

template<typename T, typename... Tail> struct is_in<T, std::tuple<T, Tail...>>
{
	static constexpr bool value = true;
};

template<typename T, typename Head, typename... Tail> struct is_in<T, std::tuple<Head, Tail...>>
{
	static constexpr bool value = is_in<T, std::tuple<Tail...>>::value;
};

static_assert(is_in<float, std::tuple<int, float>>::value);
static_assert(!is_in<float, std::tuple<int>>::value);

template<typename T, typename Args, bool is_duplicate = is_in<T, Args>::value> struct add_type
{
};

template<typename T, typename... Ts> struct add_type<T, std::tuple<Ts...>, true>
{
	using type = std::tuple<Ts...>;
};

template<typename T, typename... Ts> struct add_type<T, std::tuple<Ts...>, false>
{
	using type = std::tuple<T, Ts...>;
};

template<typename... Args> struct collapse_types;

template<> struct collapse_types<>
{
	using type = std::tuple<>;
};

template<typename Head, typename... Tail> struct collapse_types<Head, Tail...>
{
	using type = add_type<Head, typename collapse_types<Tail...>::type>::type;
};

static_assert(
	std::is_same_v<typename collapse_types<int, float, int>::type, std::tuple<float, int>>);
static_assert(std::is_same_v<typename collapse_types<std::tuple<int, float>, int>::type,
	std::tuple<std::tuple<int, float>, int>>);

template<typename> struct to_variant
{
};

template<typename... Args> struct to_variant<std::tuple<Args...>>
{
	using type = std::variant<Args...>;
};

static_assert(std::is_same_v<
	typename to_variant<typename collapse_types<std::tuple<int, float>, int>::type>::type,
	std::variant<std::tuple<int, float>, int>>);
}// namespace detail

template<size_t TypeIdx, typename... PatternTypes> struct OrPatternV2_;

template<size_t TypeIdx, typename... PatternTypes>
struct traits<OrPatternV2_<TypeIdx, PatternTypes...>>
{
  private:
	template<typename T> struct TypeExtractor_;

	template<typename... T> struct TypeExtractor_<std::tuple<T...>>
	{
		using type = typename ::detail::collapse_types<typename traits<T>::result_type...>::type;
		using type_variant = typename ::detail::to_variant<type>::type;
	};

	using _result_types = typename TypeExtractor_<std::tuple<PatternTypes...>>::type;

	static constexpr bool same_type = equal_types<_result_types>::value;

	using _result_type = std::conditional_t<same_type,
		typename std::tuple_element_t<0, _result_types>,
		typename std::conditional_t<ast_nodes<_result_types>::value,
			std::shared_ptr<ASTNode>,
			typename TypeExtractor_<std::tuple<PatternTypes...>>::type_variant>>;

  public:
	using result_type = _result_type;
};

template<size_t TypeIdx, typename... PatternTypes> struct OrPatternV2_
{
	using ResultType = typename traits<OrPatternV2_<TypeIdx, PatternTypes...>>::result_type;

	static std::optional<ResultType> match(Parser &p)
	{
		using PatternTuple = std::tuple<PatternTypes...>;
		using Pattern = typename std::tuple_element_t<TypeIdx, PatternTuple>;
		using PatternMatcher = PatternMatchV2<Pattern>;
		static_assert(
			std::tuple_size_v<typename std::invoke_result_t<decltype(PatternMatcher::match),
				Parser &>::value_type> == 1);
		if constexpr (TypeIdx == std::tuple_size_v<PatternTuple> - 1) {
			if (auto result = PatternMatcher::match(p)) {
				return std::get<0>(*result);
			} else {
				return std::nullopt;
			}
		} else {
			if (auto result = PatternMatcher::match(p)) {
				return std::get<0>(*result);
			} else {
				return OrPatternV2_<TypeIdx + 1, PatternTypes...>::match(p);
			}
		}
	}
};

template<typename... PatternTypes> struct OrPatternV2;

template<typename... PatternTypes> struct traits<OrPatternV2<PatternTypes...>>
{
	using result_type = typename traits<OrPatternV2_<0, PatternTypes...>>::result_type;
};

template<typename... PatternTypes> struct OrPatternV2 : PatternV2<OrPatternV2<PatternTypes...>>
{
	using ResultType = typename traits<OrPatternV2<PatternTypes...>>::result_type;

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		return OrPatternV2_<0, PatternTypes...>::match(p);
	}
};

template<typename... PatternTypes> struct GroupPatternsV2;

template<typename... PatternTypes> struct traits<GroupPatternsV2<PatternTypes...>>
{
	using result_type =
		typename std::invoke_result_t<decltype(PatternMatchV2<PatternTypes...>::match),
			Parser &>::value_type;
};

template<typename... PatternTypes>
struct GroupPatternsV2 : PatternV2<GroupPatternsV2<PatternTypes...>>
{
	using ResultType = typename traits<GroupPatternsV2<PatternTypes...>>::result_type;

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		return PatternMatchV2<PatternTypes...>::match(p);
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

template<Token::TokenType...> struct SingleTokenPatternV2;

template<Token::TokenType T> struct TokenResult
{
	Token token;
};

template<Token::TokenType... t> struct traits<SingleTokenPatternV2<t...>>
{
	using result_type = typename std::conditional_t<(sizeof...(t) > 1),
		std::variant<TokenResult<t>...>,
		typename std::tuple_element_t<0, std::tuple<TokenResult<t>...>>>;
};

template<Token::TokenType... Patterns>
struct SingleTokenPatternV2 : PatternV2<SingleTokenPatternV2<Patterns...>>
{
	using ResultType = traits<SingleTokenPatternV2<Patterns...>>::result_type;

	static constexpr size_t advance_by = 1;

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		if (SingleTokenPattern_<ComposedTypes<Patterns...>>::match(p)) {
			const auto &t = p.lexer().peek_token(p.token_position());
			return t.has_value() ? std::make_optional(ResultType{ *t }) : std::nullopt;
		}
		return {};
	}
};

template<typename T> struct NotAtPatternV2;

template<typename T> struct traits<NotAtPatternV2<T>>
{
	using result_type = std::monostate;
};

template<typename... Args>
struct NotAtPatternV2<std::tuple<Args...>> : PatternV2<NotAtPatternV2<std::tuple<Args...>>>
{
	using ResultType = std::monostate;

	static constexpr size_t advance_by = 0;

	template<size_t N> static constexpr bool result_(std::string_view value)
	{
		if constexpr (N == 0) {
			const auto result = std::tuple_element_t<N, std::tuple<Args...>>::matches(value);
			return !result;
		} else {
			const auto head = std::tuple_element_t<N, std::tuple<Args...>>::matches(value);
			const auto tail = result_<N - 1>(value);
			return !head && tail;
		}
	}

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		const auto this_token = p.lexer().peek_token(p.token_position());
		std::string_view value{ this_token->start().pointer_to_program,
			this_token->end().pointer_to_program };
		if (result_<sizeof...(Args) - 1>(value)) return ResultType{};
		return {};
	}
};

template<typename T> struct NegativeLookAheadV2;

template<typename T> struct traits<NegativeLookAheadV2<T>>
{
	using result_type = std::monostate;
};

template<typename PatternType>
struct NegativeLookAheadV2 : PatternV2<NegativeLookAheadV2<PatternType>>
{
	using ResultType = typename traits<NegativeLookAheadV2<PatternType>>::result_type;

	static constexpr size_t advance_by = 0;

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		const auto start = p.token_position();
		auto result = PatternType::matches(p);
		p.token_position() = start;
		return !result.has_value() ? std::make_optional(ResultType{}) : std::nullopt;
	}
};

template<typename T> struct LookAheadV2;

template<typename T> struct traits<LookAheadV2<T>>
{
	using result_type = std::monostate;
};

template<typename PatternType> struct LookAheadV2 : PatternV2<LookAheadV2<PatternType>>
{
	using ResultType = typename traits<LookAheadV2<PatternType>>::result_type;

	static constexpr size_t advance_by = 0;
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		const auto start = p.token_position();
		auto result = PatternType::matches(p);
		p.token_position() = start;
		return result.has_value() ? std::make_optional(ResultType{}) : std::nullopt;
	}
};

template<> struct traits<struct AnyToken>
{
	using result_type = Token;
};

struct AnyToken : PatternV2<AnyToken>
{
	using ResultType = Token;

	static constexpr size_t advance_by = 1;
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		return p.lexer().peek_token(p.token_position());
	}
};


template<typename... Ts> struct is_tuple : std::false_type
{
};

template<typename... Ts> struct is_tuple<std::tuple<Ts...>> : std::true_type
{
};

template<typename T> static constexpr bool is_tuple_v = is_tuple<T>{};

template<typename lhs, typename rhs> struct AndLiteralV2;

template<typename lhs, typename rhs> struct traits<AndLiteralV2<lhs, rhs>>
{
	using result_type = Token;
};

template<typename lhs, typename rhs> struct AndLiteralV2 : PatternV2<AndLiteralV2<lhs, rhs>>
{
	using ResultType = typename traits<AndLiteralV2<lhs, rhs>>::result_type;

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

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		if (lhs::matches(p)) {
			const auto token = p.lexer().peek_token(p.token_position());
			const size_t size =
				std::distance(token->start().pointer_to_program, token->end().pointer_to_program);
			std::string_view token_sv{ token->start().pointer_to_program, size };
			if (matches_<rhs>(token_sv)) { return token; }
		}
		return {};
	}
};
template<typename lhs, typename rhs> struct AndNotLiteralV2 : PatternV2<AndNotLiteralV2<lhs, rhs>>
{
	using ResultType = std::monostate;

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

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		if (lhs::matches(p)) {
			const auto token = p.lexer().peek_token(p.token_position());
			const size_t size =
				std::distance(token->start().pointer_to_program, token->end().pointer_to_program);
			std::string_view token_sv{ token->start().pointer_to_program, size };
			if (matches_<rhs>(token_sv)) { return ResultType{}; }
		}
		return {};
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

struct AsyncKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "async"; }
};

struct AwaitKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "await"; }
};

struct BreakKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "break"; }
};

struct ClassKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "class"; }
};

struct ContinueKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "continue"; }
};

struct DefKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "def"; }
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

struct LambdaKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "lambda"; }
};

struct NonLocalKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "nonlocal"; }
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

struct YieldKeywordPattern
{
	static bool matches(std::string_view token_value) { return token_value == "yield"; }
};

using ReservedKeywords = std::tuple<AndKeywordPattern,
	AsKeywordPattern,
	AssertKeywordPattern,
	AsyncKeywordPattern,
	AwaitKeywordPattern,
	BreakKeywordPattern,
	ClassKeywordPattern,
	ContinueKeywordPattern,
	DefKeywordPattern,
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
	LambdaKeywordPattern,
	NonLocalKeywordPattern,
	NotKeywordPattern,
	OrKeywordPattern,
	PassKeywordPattern,
	RaiseKeywordPattern,
	ReturnPattern,
	TryKeywordPattern,
	WhileKeywordPattern,
	WithKeywordPattern,
	YieldKeywordPattern>;

struct StarTargetPattern;

template<> struct traits<struct StarTargetPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct NAMEPattern>
{
	using result_type = TokenResult<Token::TokenType::NAME>;
};

struct NAMEPattern : PatternV2<NAMEPattern>
{
	using ResultType = typename traits<struct NAMEPattern>::result_type;

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<NotAtPatternV2<ReservedKeywords>,
			SingleTokenPatternV2<Token::TokenType::NAME>>;
		if (auto result = pattern1::match(p)) {
			const auto [_, name_token] = *result;
			return name_token;
		}

		return {};
	}
};


template<> struct traits<struct StarTargetsTupleSeq>
{
	using result_type = std::shared_ptr<Tuple>;
};

struct StarTargetsTupleSeq : PatternV2<StarTargetsTupleSeq>
{
	using ResultType = typename traits<StarTargetsTupleSeq>::result_type;

	// star_targets_tuple_seq:
	//     | star_target (',' star_target )+ [',']
	//     | star_target ','
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<ApplyInBetweenPatternV2<StarTargetPattern,
											SingleTokenPatternV2<Token::TokenType::COMMA>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("star_target (',' star_target )+ [',']");
			auto [els, _] = *result;
			return std::make_shared<Tuple>(els,
				ContextType::STORE,
				SourceLocation{
					els.front()->source_location().start, els.back()->source_location().end });
		}
		return {};
	}
};

template<> struct traits<struct StarTargetsListSeq>
{
	using result_type = std::shared_ptr<List>;
};

struct StarTargetsListSeq : PatternV2<StarTargetsListSeq>
{
	using ResultType = typename traits<StarTargetsListSeq>::result_type;

	// star_targets_list_seq: ','.star_target+ [',']
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<ApplyInBetweenPatternV2<StarTargetPattern,
											SingleTokenPatternV2<Token::TokenType::COMMA>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			auto [els, _] = *result;
			return std::make_shared<List>(els,
				ContextType::STORE,
				SourceLocation{
					els.front()->source_location().start, els.back()->source_location().end });
		}
		return {};
	}
};

template<> struct traits<struct TargetWithStarAtomPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct StarAtomPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct StarAtomPattern : PatternV2<StarAtomPattern>
{
	using ResultType = typename traits<StarAtomPattern>::result_type;
	// star_atom:
	// | NAME
	// | '(' target_with_star_atom ')'
	// | '(' [star_targets_tuple_seq] ')'
	// | '[' [star_targets_list_seq] ']'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// NAME
		using pattern1 = PatternMatchV2<NAMEPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'(' NAME ')'");
			auto [name] = *result;
			std::string id{ name.token.start().pointer_to_program,
				name.token.end().pointer_to_program };
			return std::make_shared<Name>(
				id, ContextType::STORE, SourceLocation{ name.token.start(), name.token.end() });
		}

		// NAME
		using pattern2 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LPAREN>,
			ZeroOrOnePatternV2<TargetWithStarAtomPattern>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'(' target_with_star_atom ')'");
			auto [l, target_with_star_atom, r] = *result;
			return target_with_star_atom;
		}

		// '(' [star_targets_tuple_seq] ')'
		using pattern3 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LPAREN>,
			ZeroOrOnePatternV2<StarTargetsTupleSeq>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("'(' [star_targets_tuple_seq] ')'");
			auto [l, tpl_result, r] = *result;
			std::vector<std::shared_ptr<ASTNode>> els;
			if (tpl_result.has_value()) {
				auto tpl = *tpl_result;
				els = tpl->elements();
			}
			// create new tuple with the correct source location
			return std::make_shared<Tuple>(
				els, ContextType::STORE, SourceLocation{ l.token.start(), r.token.end() });
		}

		// '[' [star_targets_list_seq] ']'
		using pattern4 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LSQB>,
			ZeroOrOnePatternV2<StarTargetsListSeq>,
			SingleTokenPatternV2<Token::TokenType::RSQB>>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("'(' [star_targets_list_seq] ')'");
			auto [l, lst_result, r] = *result;
			std::vector<std::shared_ptr<ASTNode>> els;
			if (lst_result.has_value()) {
				auto lst = *lst_result;
				els = lst->elements();
			}
			// create new list with the correct source location
			return std::make_shared<List>(
				els, ContextType::STORE, SourceLocation{ l.token.start(), r.token.end() });
		}
		return {};
	}
};

template<> struct traits<struct TLookahead>
{
	using result_type = Token;
};

struct TLookahead : PatternV2<TLookahead>
{
	using ResultType = typename traits<TLookahead>::result_type;

	// t_lookahead: '(' | '[' | '.'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("t_lookahead");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());

		using pattern1 = PatternMatchV2<OrPatternV2<SingleTokenPatternV2<Token::TokenType::LPAREN>,
			SingleTokenPatternV2<Token::TokenType::LSQB>,
			SingleTokenPatternV2<Token::TokenType::DOT>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("t_lookahead: '(' | '[' | '.'");
			auto [token] = *result;
			return std::visit([](const auto &t) -> Token { return t.token; }, token);
		}
		return {};
	}
};

template<> struct traits<struct AtomPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct SlicesPattern>
{
	using result_type = Subscript::SliceType;
};

template<> struct traits<struct GenexPattern>
{
	using result_type = std::shared_ptr<GeneratorExp>;
};

template<> struct traits<struct ArgsPattern>
{
	using ArgsType = std::vector<std::shared_ptr<ASTNode>>;
	using KwargsType = std::vector<std::shared_ptr<Keyword>>;
	using result_type = std::pair<ArgsType, KwargsType>;
};

template<> struct traits<struct ArgumentsPattern>
{
	using result_type = typename traits<ArgsPattern>::result_type;
};

template<> struct traits<struct TPrimaryPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct TPrimaryPattern : PatternV2<TPrimaryPattern>
{
	using ResultType = traits<TPrimaryPattern>::result_type;

	// t_primary:
	// 	| t_primary '.' NAME &t_lookahead
	// 	| t_primary '[' slices ']' &t_lookahead
	// 	| t_primary genexp &t_lookahead
	// 	| t_primary '(' [arguments] ')' &t_lookahead
	// 	| atom &t_lookahead

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("t_primary");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());

		// t_primary '.' NAME &t_lookahead
		using pattern1 = PatternMatchV2<TPrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::DOT>,
			SingleTokenPatternV2<Token::TokenType::NAME>,
			LookAheadV2<TLookahead>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("t_primary '.' NAME &t_lookahead");
			auto [value, _, name_token, lookahead] = *result;
			(void)lookahead;
			std::string_view name{ name_token.token.start().pointer_to_program,
				static_cast<size_t>(name_token.token.end().pointer_to_program
									- name_token.token.start().pointer_to_program) };
			return std::make_shared<Attribute>(value,
				std::string(name),
				ContextType::LOAD,
				SourceLocation{ value->source_location().start, name_token.token.end() });
		}

		// t_primary '[' slices ']' &t_lookahead
		using pattern2 = PatternMatchV2<TPrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPatternV2<Token::TokenType::RSQB>,
			LookAheadV2<TLookahead>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("t_primary '[' slices ']' &t_lookahead");
			auto [value, l, slice, r, _] = *result;
			return std::make_shared<Subscript>(value,
				slice,
				ContextType::LOAD,
				SourceLocation{ value->source_location().start, r.token.end() });
		}

		// t_primary genexp &t_lookahead
		using pattern3 = PatternMatchV2<TPrimaryPattern, GenexPattern, LookAheadV2<TLookahead>>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("t_primary genexp &t_lookahead");
			auto [value, genexp, _] = *result;
			TODO_NO_FAIL();
		}

		// t_primary '(' [arguments] ')' &t_lookahead
		using pattern4 = PatternMatchV2<TPrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::LPAREN>,
			ZeroOrOnePatternV2<ArgumentsPattern>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>,
			LookAheadV2<TLookahead>>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("'(' [arguments] ')' &t_lookahead");
			auto [function, l, arguments, r, _] = *result;
			std::vector<std::shared_ptr<ASTNode>> args;
			std::vector<std::shared_ptr<Keyword>> kwargs;
			if (arguments.has_value()) {
				auto [args_, kwargs_] = *arguments;
				args.insert(args.end(), args_.begin(), args_.end());
				kwargs.insert(kwargs.end(), kwargs_.begin(), kwargs_.end());
			}
			return std::make_shared<Call>(function,
				args,
				kwargs,
				SourceLocation{ function->source_location().start, r.token.end() });
		}

		// atom &t_lookahead
		using pattern5 = PatternMatchV2<AtomPattern, LookAheadV2<TLookahead>>;
		if (auto result = pattern5::match(p)) {
			DEBUG_LOG("atom &t_lookahead");
			auto [atom, _] = *result;
			return atom;
		}

		return {};
	}
};

struct SlicesPattern;

struct TargetWithStarAtomPattern : PatternV2<TargetWithStarAtomPattern>
{
	using ResultType = typename traits<TargetWithStarAtomPattern>::result_type;
	// target_with_star_atom:
	// 		| t_primary '.' NAME !t_lookahead
	// 		| t_primary '[' slices ']' !t_lookahead
	// 		| star_atom
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("target_with_star_atom");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());

		// t_primary '.' NAME !t_lookahead
		using pattern1 = PatternMatchV2<TPrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::DOT>,
			SingleTokenPatternV2<Token::TokenType::NAME>,
			NegativeLookAheadV2<TLookahead>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("t_primary '.' NAME !t_lookahead");
			auto [primary, d, name_token, _] = *result;
			(void)d;
			std::string name{ name_token.token.start().pointer_to_program,
				name_token.token.end().pointer_to_program };
			return std::make_shared<Attribute>(primary,
				name,
				ContextType::STORE,
				SourceLocation{ primary->source_location().start, name_token.token.end() });
		}

		using pattern2 = PatternMatchV2<TPrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPatternV2<Token::TokenType::RSQB>,
			NegativeLookAheadV2<TLookahead>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("t_primary '[' slices ']' !t_lookahead");
			auto [value, l, subscript, r, _] = *result;
			return std::make_shared<Subscript>(value,
				subscript,
				ContextType::STORE,
				SourceLocation{ value->source_location().start, r.token.end() });
		}

		// star_atom
		using pattern3 = PatternMatchV2<StarAtomPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("star_atom");
			auto [star_atom] = *result;
			return star_atom;
		}
		return {};
	}
};


struct StarTargetPattern : PatternV2<StarTargetPattern>
{
	using ResultType = std::shared_ptr<ASTNode>;
	// star_target:
	//     | '*' (!'*' star_target)
	//     | target_with_star_atom
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("star_target");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());

		// '*' (!'*' star_target)
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::STAR>,
			NegativeLookAheadV2<SingleTokenPatternV2<Token::TokenType::STAR>>,
			StarTargetPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'*' (!'*' star_target)");
			(void)result;
			TODO_NO_FAIL();
		}

		// target_with_star_atom
		using pattern2 = PatternMatchV2<TargetWithStarAtomPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("target_with_star_atom");
			auto [target] = *result;
			return target;
		}
		return {};
	}
};

template<> struct traits<struct StarTargetsPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct StarTargetsPattern : PatternV2<StarTargetsPattern>
{
	using ResultType = typename traits<StarTargetsPattern>::result_type;

	// star_targets:
	// | star_target !','
	// | star_target (',' star_target )* [',']
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("star_targets");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());

		// star_target !','
		using pattern1 = PatternMatchV2<StarTargetPattern,
			NegativeLookAheadV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("star_target !','");
			auto [star_target, _] = *result;
			return star_target;
		}

		// star_target (',' star_target )* [',']
		using pattern2 = PatternMatchV2<StarTargetPattern,
			ZeroOrMorePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>, StarTargetPattern>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("star_target (',' star_target )* [',']");
			auto [target, targets_, _] = *result;
			std::vector<std::shared_ptr<ASTNode>> targets;
			targets.reserve(1 + targets_.size());
			targets.push_back(target);
			std::transform(
				targets_.begin(), targets_.end(), std::back_inserter(targets), [](auto &&el) {
					auto [_, target] = el;
					return target;
				});
			return std::make_shared<Tuple>(targets,
				ContextType::STORE,
				SourceLocation{ targets.front()->source_location().start,
					targets.back()->source_location().end });
		}

		return {};
	}
};

std::shared_ptr<Constant> parse_bytes(Token string)
{
	Bytes byte_collection;
	auto *start = string.start().pointer_to_program;
	const auto *end = string.end().pointer_to_program;

	if (start[0] != 'b' && start[0] != 'B') { return nullptr; }
	start++;

	const bool is_triple_quote = (end - start) >= 3 && (start[0] == '\"' || start[0] == '\'')
								 && (start[1] == '\"' || start[1] == '\'')
								 && (start[2] == '\"' || start[2] == '\'');

	const auto value = [is_triple_quote, start, end]() {
		if (is_triple_quote) {
			return std::string{ start + 3, end - 3 };
		} else {
			return std::string{ start + 1, end - 1 };
		}
	}();
	return std::make_shared<Constant>(
		Bytes::from_unescaped_string(value), SourceLocation{ string.start(), string.end() });
}

template<> struct traits<struct FStringReplacementFieldPattern>
{
	using result_type = std::pair<std::shared_ptr<FormattedValue>, std::shared_ptr<Constant>>;
};

template<> struct traits<struct FStringFormatSpecPattern>
{
	using result_type = std::variant<typename traits<FStringReplacementFieldPattern>::result_type,
		std::shared_ptr<Constant>>;
};

template<> struct traits<struct FExpressionPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct YieldExpressionPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct StarExpressionsPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct FExpressionPattern : PatternV2<FExpressionPattern>
{
	using ResultType = typename traits<FExpressionPattern>::result_type;

	// f_expression
	//     | (yield_expr | star_expressions)
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatchV2<OrPatternV2<YieldExpressionPattern, StarExpressionsPattern>>;
		if (auto result = pattern1::match(p)) {
			auto [fexpression] = *result;
			return fexpression;
		}
		return {};
	}
};

template<> struct traits<struct ConversionPattern>
{
	using result_type = FormattedValue::Conversion;
};

struct ConversionPattern : PatternV2<ConversionPattern>
{
	using ResultType = typename traits<ConversionPattern>::result_type;

	// conversion
	//     | ("s" | "r" | "a")
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::NAME>>;
		if (auto result = pattern1::match(p)) {
			auto [conversion] = *result;
			std::string_view s{ conversion.token.start().pointer_to_program,
				conversion.token.end().pointer_to_program };
			if (s.size() == 1) { return to_formatted_value_conversion(s[0]); }
		}

		return {};
	}

	static std::optional<FormattedValue::Conversion> to_formatted_value_conversion(const char c)
	{
		switch (c) {
		case 'r': {
			return FormattedValue::Conversion::REPR;
		} break;
		case 's': {
			return FormattedValue::Conversion::STRING;
		} break;
		case 'a': {
			return FormattedValue::Conversion::ASCII;
		} break;
		}
		return {};
	}
};


struct FStringReplacementFieldPattern : PatternV2<FStringReplacementFieldPattern>
{
	using ResultType = typename traits<FStringReplacementFieldPattern>::result_type;

	// fstring_replacement_field
	//     | '{' f_expression "="? [ "!" conversion ] [ ':' fstring_format_spec* ] '}'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LBRACE>,
			FExpressionPattern,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::EQUAL>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::EXCLAMATION>,
				ConversionPattern>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COLON>,
				ZeroOrMorePatternV2<FStringFormatSpecPattern>>,
			SingleTokenPatternV2<Token::TokenType::RBRACE>>;
		if (auto result = pattern1::match(p)) {
			auto [l, expression, display, conversion, format_spec, r] = *result;
			auto c = [conversion = std::move(conversion)]() {
				if (conversion.has_value()) { return std::get<1>(*conversion); }
				return FormattedValue::Conversion::NONE;
			}();
			const auto next = p.lexer().peek_token(p.token_position());
			if (next.has_value() && next->token_type() != Token::TokenType::ENDMARKER) {
				std::string value{ r.token.end().pointer_to_program,
					next->start().pointer_to_program };
				if (!value.empty()) {
					return { { std::make_shared<FormattedValue>(expression,
								   c,
								   nullptr,
								   SourceLocation{ l.token.start(), l.token.end() }),
						std::make_shared<Constant>(value, SourceLocation{}) } };
				}
			}
			return {
				{ std::make_shared<FormattedValue>(
					  expression, c, nullptr, SourceLocation{ l.token.start(), l.token.end() }),
					nullptr }
			};
		}

		return {};
	}
};

template<> struct traits<struct FStringMiddlePattern>
{
	using result_type = std::variant<typename traits<FStringReplacementFieldPattern>::result_type,
		std::shared_ptr<Constant>>;
};

struct FStringMiddlePattern : PatternV2<FStringMiddlePattern>
{
	// fstring_middle
	//     | fstring_replacement_field
	//     | FSTRING_MIDDLE
	using ResultType = typename traits<FStringMiddlePattern>::result_type;

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<FStringReplacementFieldPattern>;
		if (auto result = pattern1::match(p)) {
			auto [fstring_replacement_field] = *result;
			return fstring_replacement_field;
		}

		using pattern2 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::FSTRING_MIDDLE>>;
		if (auto result = pattern2::match(p)) {
			auto [middle] = *result;
			std::string_view middle_str{ middle.token.start().pointer_to_program,
				middle.token.end().pointer_to_program };
			std::string str;
			str.reserve(middle_str.size());
			for (size_t i = 0; i < middle_str.size(); ++i) {
				if ((i + 1) < middle_str.size() && middle_str[i] == '{'
					&& middle_str[i + 1] == '{') {
					str.push_back('{');
					i++;
				} else if ((i + 1) < middle_str.size() && middle_str[i] == '}'
						   && middle_str[i + 1] == '}') {
					str.push_back('}');
					i++;
				} else {
					str.push_back(middle_str[i]);
				}
			}
			return std::make_shared<Constant>(
				std::move(str), SourceLocation{ middle.token.start(), middle.token.end() });
		}

		return {};
	}
};

struct FStringFormatSpecPattern : PatternV2<FStringFormatSpecPattern>
{
	using ResultType = typename traits<FStringFormatSpecPattern>::result_type;

	// fstring_format_spec:
	//     | FSTRING_MIDDLE
	//     | fstring_replacement_field
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::FSTRING_MIDDLE>>;
		if (auto result = pattern1::match(p)) {
			auto [middle] = *result;
			std::string middle_str{ middle.token.start().pointer_to_program,
				middle.token.end().pointer_to_program };
			return std::make_shared<Constant>(
				std::move(middle_str), SourceLocation{ middle.token.start(), middle.token.end() });
		}

		using pattern2 = PatternMatchV2<FStringReplacementFieldPattern>;
		if (auto result = pattern2::match(p)) {
			auto [fstring_replacement_field] = *result;
			return fstring_replacement_field;
		}

		return {};
	}
};

template<> struct traits<struct FStringPattern>
{
	using result_type = std::shared_ptr<JoinedStr>;
};

struct FStringPattern : PatternV2<FStringPattern>
{
	// fstring
	//     | FSTRING_START fstring_middle* FSTRING_END
	using ResultType = typename traits<FStringPattern>::result_type;

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// FSTRING_START fstring_middle* FSTRING_END
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::FSTRING_START>,
			ZeroOrMorePatternV2<FStringMiddlePattern>,
			SingleTokenPatternV2<Token::TokenType::FSTRING_END>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("FSTRING_START fstring_middle* FSTRING_END");
			auto [start, middle, end] = *result;
			std::vector<std::shared_ptr<ASTNode>> string_nodes;
			string_nodes.reserve(middle.size());
			for (const auto &n : middle) {
				std::visit(overloaded{
							   [&string_nodes](const std::shared_ptr<Constant> &c) {
								   string_nodes.push_back(c);
							   },
							   [&string_nodes](const std::pair<std::shared_ptr<FormattedValue>,
								   std::shared_ptr<Constant>> &p) {
								   auto [fv, c] = p;
								   string_nodes.push_back(fv);
								   if (c) { string_nodes.push_back(c); }
							   },
						   },
					n);
			}
			if (string_nodes.empty()) {
				string_nodes.push_back(std::make_shared<Constant>(
					std::string{}, SourceLocation{ start.token.end(), end.token.start() }));
			}
			return std::make_shared<JoinedStr>(
				std::move(string_nodes), SourceLocation{ start.token.start(), end.token.end() });
		}
		return {};
	}
};

template<> struct traits<struct StringPattern>
{
	using result_type = std::shared_ptr<Constant>;
};

struct StringPattern : PatternV2<StringPattern>
{
	// STRING
	using ResultType = typename traits<StringPattern>::result_type;

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::STRING>>;

		if (auto result = pattern1::match(p)) {
			auto [str] = *result;
			if (auto c = parse_bytes(str.token)) { return c; }

			std::string_view value{ str.token.start().pointer_to_program,
				str.token.end().pointer_to_program };

			const bool is_triple_quote = [value]() {
				if (value.size() < 3) { return false; }
				return (value[0] == '\"' || value[0] == '\'')
					   && (value[1] == '\"' || value[1] == '\'')
					   && (value[2] == '\"' || value[2] == '\'');
			}();

			if (is_triple_quote) {
				value = value.substr(3, value.size() - 6);
			} else {
				value = value.substr(1, value.size() - 2);
			}
			return std::make_shared<Constant>(String::from_unescaped_string(std::string{ value }),
				SourceLocation{ str.token.start(), str.token.end() });
		}
		return {};
	}
};

template<> struct traits<struct StringsPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct StringsPattern : PatternV2<StringsPattern>
{
	using ResultType = traits<StringsPattern>::result_type;
	// strings: (fstring|string)+
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatchV2<OneOrMorePatternV2<OrPatternV2<FStringPattern, StringPattern>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("strings: STRING+");
			auto [strings] = *result;
			std::vector<std::shared_ptr<ASTNode>> string_nodes;
			string_nodes.reserve(strings.size());
			for (const auto &el : strings) {
				if (auto joined_str = as<JoinedStr>(el)) {
					string_nodes.insert(string_nodes.end(),
						joined_str->values().begin(),
						joined_str->values().end());
				} else if (auto c = as<Constant>(el)) {
					string_nodes.push_back(c);
				} else {
					TODO();
				}
			}
			SourceLocation sl{ string_nodes.front()->source_location().start,
				string_nodes.back()->source_location().end };

			bool all_constant = std::all_of(string_nodes.begin(),
				string_nodes.end(),
				[](const auto &el) -> bool { return static_cast<bool>(as<Constant>(el)); });
			if (all_constant) {
				if (std::holds_alternative<Bytes>(*as<Constant>(string_nodes.front())->value())) {
					Bytes bytes;
					for (const auto &el : string_nodes) {
						ASSERT(as<Constant>(el));
						auto c = as<Constant>(el);
						if (!std::holds_alternative<Bytes>(*c->value())) {
							std::cerr << "SyntaxError: cannot mix bytes and nonbytes literals\n";
							std::abort();
						}
						const auto &byte = std::get<Bytes>(*c->value());
						bytes.b.insert(bytes.b.end(), byte.b.begin(), byte.b.end());
					}
					return std::make_shared<Constant>(bytes, sl);
				} else {
					auto str = std::accumulate(string_nodes.begin(),
						string_nodes.end(),
						std::string{},
						[](std::string acc, const auto &el) {
							ASSERT(as<Constant>(el));
							auto c = as<Constant>(el);
							ASSERT(std::holds_alternative<String>(*c->value()));
							return acc + std::get<String>(*c->value()).s;
						});
					return std::make_shared<Constant>(std::move(str), sl);
				}
			}
			return std::make_shared<JoinedStr>(std::move(string_nodes), sl);
		}
		return {};
	}
};

template<> struct traits<struct BitwiseOrPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct NamedExpressionPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct StarNamedExpression>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct StarNamedExpression : PatternV2<StarNamedExpression>
{
	using ResultType = typename traits<StarNamedExpression>::result_type;
	// star_named_expression:
	// | '*' bitwise_or
	// | named_expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::STAR>, BitwiseOrPattern>;
		if (auto result = pattern1::match(p)) {
			auto [star_token, value] = *result;
			return std::make_shared<Starred>(value,
				ContextType::LOAD,
				SourceLocation{ star_token.token.start(), value->source_location().start });
		}

		using pattern2 = PatternMatchV2<NamedExpressionPattern>;
		if (auto result = pattern2::match(p)) {
			auto [name_expression] = *result;
			return name_expression;
		}

		return {};
	}
};

template<> struct traits<struct StarNamedExpressions>
{
	using result_type = std::vector<std::shared_ptr<ASTNode>>;
};

struct StarNamedExpressions : PatternV2<StarNamedExpressions>
{
	using ResultType = typename traits<StarNamedExpressions>::result_type;

	// star_named_expressions: ','.star_named_expression+ [',']
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("star_named_expressions");
		using pattern1 = PatternMatchV2<ApplyInBetweenPatternV2<StarNamedExpression,
											SingleTokenPatternV2<Token::TokenType::COMMA>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			auto [named_expressions, _] = *result;
			return named_expressions;
		}

		return {};
	}
};

template<> struct traits<struct ListPattern>
{
	using result_type = std::shared_ptr<List>;
};

struct ListPattern : PatternV2<ListPattern>
{
	using ResultType = typename traits<ListPattern>::result_type;

	// list: '[' [star_named_expressions] ']'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("list");
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LSQB>,
			ZeroOrOnePatternV2<StarNamedExpressions>,
			SingleTokenPatternV2<Token::TokenType::RSQB>>;
		if (auto result = pattern1::match(p)) {
			auto [l, els, r] = *result;
			return std::make_shared<List>(els.value_or(std::vector<std::shared_ptr<ASTNode>>{}),
				ContextType::LOAD,
				SourceLocation{ l.token.start(), r.token.end() });
		}
		return {};
	}
};

template<> struct traits<struct DisjunctionPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct ForIfClausePattern>
{
	using result_type = std::shared_ptr<Comprehension>;
};

struct ForIfClausePattern : PatternV2<ForIfClausePattern>
{
	using ResultType = typename traits<ForIfClausePattern>::result_type;

	// for_if_clause:
	//     | ASYNC 'for' star_targets 'in' ~ disjunction ('if' disjunction )*
	//     | 'for' star_targets 'in' ~ disjunction ('if' disjunction )*
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// using pattern1 = PatternMatch<OneOrMorePattern<ForIfClausePattern>>;
		// if (pattern1::match(p)) { return true; }

		// TODO: implement commit pattern (~)
		using pattern2 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ForKeywordPattern>,
			StarTargetsPattern,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, InKeywordPattern>,
			DisjunctionPattern,
			ZeroOrMorePatternV2<
				AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, IfKeywordPattern>,
				DisjunctionPattern>>;
		if (auto result = pattern2::match(p)) {
			auto [for_token, target, _, iter, ifs_group] = *result;
			std::vector<std::shared_ptr<ASTNode>> ifs;
			ifs.reserve(ifs_group.size());
			std::transform(
				ifs_group.begin(), ifs_group.end(), std::back_inserter(ifs), [](const auto &el) {
					auto [_, if_] = el;
					return if_;
				});
			SourceLocation sc{ .start = for_token.start(),
				.end =
					ifs.empty() ? iter->source_location().end : ifs.back()->source_location().end };
			return std::make_shared<Comprehension>(target, iter, ifs, false, sc);
		}
		return {};
	}
};

template<> struct traits<struct ForIfClausesPattern>
{
	using result_type = std::vector<std::shared_ptr<Comprehension>>;
};

struct ForIfClausesPattern : PatternV2<ForIfClausesPattern>
{
	using ResultType = typename traits<ForIfClausesPattern>::result_type;

	// for_if_clauses:
	//     | for_if_clause+
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<OneOrMorePatternV2<ForIfClausePattern>>;
		if (auto result = pattern1::match(p)) {
			auto [for_if_clauses] = *result;
			return for_if_clauses;
		}
		return {};
	}
};

template<> struct traits<struct ListCompPattern>
{
	using result_type = std::shared_ptr<ListComp>;
};


struct ListCompPattern : PatternV2<ListCompPattern>
{
	using ResultType = typename traits<ListCompPattern>::result_type;
	// listcomp:
	//     | '[' named_expression ~ for_if_clauses ']'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// TODO: support commit pattern (~)
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LSQB>,
			NamedExpressionPattern,
			ForIfClausesPattern,
			SingleTokenPatternV2<Token::TokenType::RSQB>>;
		if (auto result = pattern1::match(p)) {
			auto [l, elt, generators, r] = *result;
			SourceLocation sc{
				.start = l.token.start(),
				.end = r.token.end(),
			};
			return std::make_shared<ListComp>(elt, std::move(generators), sc);
		}
		return {};
	}
};

template<> struct traits<struct TuplePattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct TuplePattern : PatternV2<TuplePattern>
{
	using ResultType = typename traits<TuplePattern>::result_type;

	// tuple:
	//     | '(' [star_named_expression ',' [star_named_expressions]  ] ')'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("tuple");
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LPAREN>,
			ZeroOrOnePatternV2<StarNamedExpression,
				SingleTokenPatternV2<Token::TokenType::COMMA>,
				ZeroOrOnePatternV2<StarNamedExpressions>>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>>;
		if (auto result = pattern1::match(p)) {
			auto [l, maybe_named_expression, r] = *result;

			if (!maybe_named_expression.has_value()) {
				return std::make_shared<Tuple>(std::vector<std::shared_ptr<ASTNode>>{},
					ContextType::LOAD,
					SourceLocation{ l.token.start(), r.token.end() });
			}
			std::vector<std::shared_ptr<ASTNode>> elements;
			const auto &named_expression = *maybe_named_expression;
			auto [lhs, _, els_] = named_expression;
			auto els = els_.value_or(std::vector<std::shared_ptr<ASTNode>>{});
			elements.reserve(els.size() + 1);
			elements.push_back(lhs);
			elements.insert(elements.end(), els.begin(), els.end());
			return std::make_shared<Tuple>(
				elements, ContextType::LOAD, SourceLocation{ l.token.start(), r.token.end() });
		}
		return {};
	}
};

template<> struct traits<struct GroupPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct GroupPattern : PatternV2<GroupPattern>
{
	using ResultType = typename traits<GroupPattern>::result_type;

	// group:
	//     | '(' (yield_expr | named_expression) ')'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("group");

		// TODO: add yield_expr
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LPAREN>,
			OrPatternV2<YieldExpressionPattern, NamedExpressionPattern>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'(' (yield_expr | named_expression) ')'");
			return std::get<1>(*result);
		}
		return {};
	}
};

struct GenexPattern : PatternV2<GenexPattern>
{
	using ResultType = typename traits<GenexPattern>::result_type;

	// genexp:
	//     | '(' named_expression ~ for_if_clauses ')'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// TODO: implement commit (~) pattern
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LPAREN>,
			NamedExpressionPattern,
			ForIfClausesPattern,
			SingleTokenPatternV2<Token::TokenType::RPAREN>>;
		if (auto result = pattern1::match(p)) {
			auto [l, expression, generators, r] = *result;
			SourceLocation sc{
				.start = l.token.start(),
				.end = r.token.end(),
			};
			return std::make_shared<GeneratorExp>(expression, std::move(generators), sc);
		}
		return {};
	}
};

template<> struct traits<struct ExpressionPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct KVPairPattern>
{
	using result_type = std::pair<typename traits<ExpressionPattern>::result_type,
		typename traits<ExpressionPattern>::result_type>;
};

struct KVPairPattern : PatternV2<KVPairPattern>
{
	using ResultType = typename traits<KVPairPattern>::result_type;

	// kvpair: expression ':' expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("kvpair");
		using pattern1 = PatternMatchV2<ExpressionPattern,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			ExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			auto [key, _, value] = *result;
			return { { key, value } };
		}
		return {};
	}
};

template<> struct traits<struct DoubleStarredKVPairPattern>
{
	using result_type = typename traits<KVPairPattern>::result_type;
};

struct DoubleStarredKVPairPattern : PatternV2<DoubleStarredKVPairPattern>
{
	using ResultType = typename traits<DoubleStarredKVPairPattern>::result_type;

	// double_starred_kvpair:
	// 		| '**' bitwise_or
	// 		| kvpair
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("double_starred_kvpair");
		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::DOUBLESTAR>, BitwiseOrPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'**' bitwise_or");
			auto [_, value] = *result;
			return { { nullptr, value } };
		}

		using pattern2 = PatternMatchV2<KVPairPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("kvpair");
			auto [kvpair] = *result;
			return kvpair;
		}

		return {};
	}
};

template<> struct traits<struct DoubleStarredKVPairsPattern>
{
	using result_type = std::vector<typename traits<DoubleStarredKVPairPattern>::result_type>;
};

struct DoubleStarredKVPairsPattern : PatternV2<DoubleStarredKVPairsPattern>
{
	using ResultType = typename traits<DoubleStarredKVPairsPattern>::result_type;

	// double_starred_kvpairs: ','.double_starred_kvpair+ [',']
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("double_starred_kvpairs");
		using pattern1 = PatternMatchV2<ApplyInBetweenPatternV2<DoubleStarredKVPairPattern,
											SingleTokenPatternV2<Token::TokenType::COMMA>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("','.double_starred_kvpair+ [',']");
			auto [kvpairs, _] = *result;
			return kvpairs;
		}
		return {};
	}
};

template<> struct traits<struct DictPattern>
{
	using result_type = std::shared_ptr<Dict>;
};

struct DictPattern : PatternV2<DictPattern>
{
	using ResultType = typename traits<DictPattern>::result_type;

	// dict:
	// | '{' [double_starred_kvpairs] '}'
	// | '{' invalid_double_starred_kvpairs '}'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// '{' [double_starred_kvpairs] '}'
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LBRACE>,
			ZeroOrOnePatternV2<DoubleStarredKVPairsPattern>,
			SingleTokenPatternV2<Token::TokenType::RBRACE>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'{' [double_starred_kvpairs] '}'");

			auto [l, kv_pairs, r] = *result;
			std::vector<std::shared_ptr<ASTNode>> keys;
			std::vector<std::shared_ptr<ASTNode>> values;

			if (kv_pairs.has_value()) {
				ASSERT(!kv_pairs->empty())
				keys.reserve(kv_pairs->size());
				values.reserve(kv_pairs->size());
				for (const auto &[k, v] : *kv_pairs) {
					keys.push_back(k);
					values.push_back(v);
				}
			}

			auto dict = std::make_shared<Dict>(
				keys, values, SourceLocation{ l.token.start(), r.token.end() });
			return dict;
		}

		return {};
	}
};

template<> struct traits<struct SetPattern>
{
	using result_type = std::shared_ptr<Set>;
};

struct SetPattern : PatternV2<SetPattern>
{
	using ResultType = typename traits<SetPattern>::result_type;

	// set: '{' star_named_expressions '}'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LBRACE>,
			StarNamedExpressions,
			SingleTokenPatternV2<Token::TokenType::RBRACE>>;
		if (auto result = pattern1::match(p)) {
			auto [l, set_values, r] = *result;
			SourceLocation sl{
				.start = l.token.start(),
				.end = r.token.end(),
			};
			return std::make_shared<Set>(std::move(set_values), ContextType::LOAD, sl);
		}
		return {};
	}
};

template<> struct traits<struct DictCompPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct DictCompPattern : PatternV2<DictCompPattern>
{
	using ResultType = typename traits<DictCompPattern>::result_type;

	// dictcomp: '{' kvpair for_if_clauses '}'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LBRACE>,
			KVPairPattern,
			ForIfClausesPattern,
			SingleTokenPatternV2<Token::TokenType::RBRACE>>;
		if (auto result = pattern1::match(p)) {
			auto [l, kvpair, generators, r] = *result;
			SourceLocation sc{
				.start = l.token.start(),
				.end = r.token.end(),
			};
			auto [key, value] = kvpair;
			return std::make_shared<DictComp>(key, value, std::move(generators), sc);
		}
		return {};
	}
};

template<> struct traits<struct SetCompPattern>
{
	using result_type = std::shared_ptr<SetComp>;
};

struct SetCompPattern : PatternV2<SetCompPattern>
{
	using ResultType = typename traits<SetCompPattern>::result_type;

	// setcomp:
	//     | '{' named_expression ~ for_if_clauses '}'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// TODO: implement commit pattern (~)
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LBRACE>,
			NamedExpressionPattern,
			ForIfClausesPattern,
			SingleTokenPatternV2<Token::TokenType::RBRACE>>;
		if (auto result = pattern1::match(p)) {
			auto [l, elt, generators, r] = *result;
			SourceLocation sc{
				.start = l.token.start(),
				.end = r.token.end(),
			};
			return std::make_shared<SetComp>(elt, std::move(generators), sc);
		}
		return {};
	}
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
		std::istringstream is(value);
		double float_value;
		is >> float_value;
		return std::make_shared<Constant>(float_value, source_location);
	} else if (value.find('.') != std::string::npos) {
		// float
		double float_value = std::stod(value);
		return std::make_shared<Constant>(float_value, source_location);
	} else {
		// int
		mpz_class big_int{ value };
		if (big_int.fits_slong_p()) {
			return std::make_shared<Constant>(big_int.get_si(), source_location);
		} else {
			return std::make_shared<Constant>(big_int, source_location);
		}
	}
}


struct AtomPattern : PatternV2<AtomPattern>
{
	using ResultType = typename traits<AtomPattern>::result_type;
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
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("atom");
		{
			const auto token = p.lexer().peek_token(p.token_position());
			std::string name{ token->start().pointer_to_program, token->end().pointer_to_program };
			DEBUG_LOG(name);
		}
		// NAME
		using pattern1 = PatternMatchV2<NAMEPattern>;
		if (auto name_result = pattern1::match(p)) {
			DEBUG_LOG("NAME");

			auto [token] = *name_result;
			std::string name{ token.token.start().pointer_to_program,
				token.token.end().pointer_to_program };
			if (name == "True") {
				return std::make_shared<Constant>(
					true, SourceLocation{ token.token.start(), token.token.end() });
			} else if (name == "False") {
				return std::make_shared<Constant>(
					false, SourceLocation{ token.token.start(), token.token.end() });
			} else if (name == "None") {
				return std::make_shared<Constant>(py::NameConstant{ py::NoneType{} },
					SourceLocation{ token.token.start(), token.token.end() });
			} else {
				return std::make_shared<Name>(name,
					ContextType::LOAD,
					SourceLocation{ token.token.start(), token.token.end() });
			}
		}
		// strings
		using pattern6 = PatternMatchV2<StringsPattern>;
		if (auto result = pattern6::match(p)) {
			DEBUG_LOG("strings");
			auto [strings] = *result;
			return strings;
		}

		// NUMBER
		using pattern7 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::NUMBER>>;
		if (auto result = pattern7::match(p)) {
			DEBUG_LOG("NUMBER");
			auto [token] = *result;
			std::string number{ token.token.start().pointer_to_program,
				token.token.end().pointer_to_program };
			return parse_number(number, SourceLocation{ token.token.start(), token.token.end() });
		}

		// 	| (tuple | group | genexp)
		using pattern8 = PatternMatchV2<OrPatternV2<TuplePattern, GroupPattern, GenexPattern>>;
		if (auto result = pattern8::match(p)) {
			DEBUG_LOG("(tuple | group | genexp)");
			auto [tuple_or_group_genexp] = *result;
			return tuple_or_group_genexp;
		}

		// 	| (list | listcomp)
		using pattern9 = PatternMatchV2<OrPatternV2<ListPattern, ListCompPattern>>;
		if (auto result = pattern9::match(p)) {
			DEBUG_LOG("(list | listcomp)");
			auto [list_or_listcomp] = *result;
			return list_or_listcomp;
		}

		// (dict | set | dictcomp | setcomp)
		using pattern10 =
			PatternMatchV2<OrPatternV2<DictPattern, SetPattern, DictCompPattern, SetCompPattern>>;
		if (auto result = pattern10::match(p)) {
			DEBUG_LOG("(dict | set | dictcomp | setcomp)");
			auto [dict_or_set_or_dictcomp_or_setcomp] = *result;
			return dict_or_set_or_dictcomp_or_setcomp;
		}

		using pattern11 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::ELLIPSIS>>;
		if (auto result = pattern11::match(p)) {
			auto [token] = *result;
			return std::make_shared<Constant>(
				Ellipsis{}, SourceLocation{ token.token.start(), token.token.end() });
		}

		return {};
	}
};


struct NamedExpressionPattern : PatternV2<NamedExpressionPattern>
{
	using ResultType = typename traits<NamedExpressionPattern>::result_type;
	// named_expression:
	// 	| NAME ':=' ~ expression
	// 	| expression !':='
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("NamedExpressionPattern");
		const auto token = p.lexer().peek_token(p.token_position());
		DEBUG_LOG("{}", token->to_string());
		std::string_view maybe_name{ token->start().pointer_to_program,
			static_cast<size_t>(
				token->end().pointer_to_program - token->start().pointer_to_program) };

		// NAME ':=' ~ expression
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::NAME>,
			SingleTokenPatternV2<Token::TokenType::COLONEQUAL>,
			ExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("NAME ':=' ~ expression");
			auto [name_token, _, expr] = *result;

			std::string name{ name_token.token.start().pointer_to_program,
				name_token.token.end().pointer_to_program };

			auto target = std::make_shared<Name>(name,
				ContextType::STORE,
				SourceLocation{ name_token.token.start(), name_token.token.end() });
			return std::make_shared<NamedExpr>(target,
				expr,
				SourceLocation{ target->source_location().start, expr->source_location().end });
		}


		// expression !':='
		using pattern2 = PatternMatchV2<ExpressionPattern,
			NegativeLookAheadV2<SingleTokenPatternV2<Token::TokenType::COLONEQUAL>>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("expression !':='");
			auto [expression, _] = *result;
			return expression;
		}
		return {};
	}
};

template<> struct traits<struct StarredExpressionPattern>
{
	using result_type = std::shared_ptr<Starred>;
};

template<> struct traits<struct KwargsOrStarredPattern>
{
	using result_type = std::shared_ptr<Keyword>;
};

struct KwargsOrStarredPattern : PatternV2<KwargsOrStarredPattern>
{
	using ResultType = typename traits<KwargsOrStarredPattern>::result_type;

	// kwarg_or_starred:
	//     | NAME '=' expression
	//     | starred_expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("kwarg_or_starred");

		// NAME '=' expression
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::NAME>,
			SingleTokenPatternV2<Token::TokenType::EQUAL>,
			ExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("NAME '=' expression");
			auto [name_token, _, expression] = *result;
			std::string name{ name_token.token.start().pointer_to_program,
				name_token.token.end().pointer_to_program };
			return std::make_shared<Keyword>(name,
				expression,
				SourceLocation{ name_token.token.start(), expression->source_location().end });
		}

		// starred_expression
		using pattern2 = PatternMatchV2<StarredExpressionPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("starred_expression");
			auto [expression] = *result;
			return std::make_shared<Keyword>(expression,
				SourceLocation{
					expression->source_location().start, expression->source_location().end });
		}
		return {};
	}
};

struct StarredExpressionPattern : PatternV2<StarredExpressionPattern>
{
	using ResultType = typename traits<StarredExpressionPattern>::result_type;

	// starred_expression:
	//     | '*' expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("starred_expression");

		// '*' expression
		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::STAR>, ExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'*' expression");
			auto [token, expression] = *result;
			return std::make_shared<Starred>(expression,
				ContextType::LOAD,
				SourceLocation{ token.token.start(), expression->source_location().end });
		}
		return {};
	}
};

template<> struct traits<struct KwargsOrDoubleStarredPattern>
{
	using result_type = std::shared_ptr<Keyword>;
};

struct KwargsOrDoubleStarredPattern : PatternV2<KwargsOrDoubleStarredPattern>
{
	using ResultType = typename traits<KwargsOrDoubleStarredPattern>::result_type;

	// kwarg_or_double_starred:
	//     | NAME '=' expression
	//     | '**' expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("kwarg_or_double_starred");

		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::NAME>,
			SingleTokenPatternV2<Token::TokenType::EQUAL>,
			ExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("NAME '=' expression");
			auto [name_token, _, expression] = *result;
			std::string name{ name_token.token.start().pointer_to_program,
				name_token.token.end().pointer_to_program };
			return std::make_shared<Keyword>(name,
				expression,
				SourceLocation{ name_token.token.start(), expression->source_location().end });
		}

		using pattern2 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::DOUBLESTAR>, ExpressionPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'**' expression");
			auto [token, expression] = *result;
			return std::make_shared<Keyword>(expression,
				SourceLocation{ token.token.start(), expression->source_location().end });
		}

		return {};
	}
};

template<> struct traits<struct KwargsPattern>
{
	using result_type = std::vector<std::shared_ptr<Keyword>>;
};

struct KwargsPattern : PatternV2<KwargsPattern>
{
	using ResultType = typename traits<KwargsPattern>::result_type;

	// kwargs:
	//     | ','.kwarg_or_starred+ ',' ','.kwarg_or_double_starred+
	//     | ','.kwarg_or_starred+
	//     | ','.kwarg_or_double_starred+
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("kwargs");

		// ','.kwarg_or_starred+ ',' ','.kwarg_or_double_starred+
		using pattern1 = PatternMatchV2<ApplyInBetweenPatternV2<KwargsOrStarredPattern,
											SingleTokenPatternV2<Token::TokenType::COMMA>>,
			SingleTokenPatternV2<Token::TokenType::COMMA>,
			ApplyInBetweenPatternV2<KwargsOrDoubleStarredPattern,
				SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("','.kwarg_or_starred+ ',' ','.kwarg_or_double_starred+");
			auto [kwarg_or_starred, _, kwarg_or_double_starred] = *result;
			ResultType kwargs;
			kwargs.reserve(kwarg_or_starred.size() + kwarg_or_double_starred.size());
			kwargs.insert(kwargs.end(), kwarg_or_starred.begin(), kwarg_or_starred.end());
			kwargs.insert(
				kwargs.end(), kwarg_or_double_starred.begin(), kwarg_or_double_starred.end());
			return kwargs;
		}

		// ','.kwarg_or_starred+
		using pattern2 = PatternMatchV2<ApplyInBetweenPatternV2<KwargsOrStarredPattern,
			SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("','.kwarg_or_starred+");
			auto [kwarg_or_starred] = *result;
			return kwarg_or_starred;
		}

		// ','.kwarg_or_double_starred+
		using pattern3 = PatternMatchV2<ApplyInBetweenPatternV2<KwargsOrDoubleStarredPattern,
			SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("','.kwarg_or_double_starred+");
			auto [kwarg_or_double_starred] = *result;
			return kwarg_or_double_starred;
		}
		return {};
	}
};

struct ArgsPattern : PatternV2<ArgsPattern>
{
	using ResultType = typename traits<ArgsPattern>::result_type;

	// args:
	// 	| ','.(starred_expression | named_expression !'=')+ [',' kwargs ]
	// 	| kwargs
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("args");
		using pattern1 = PatternMatchV2<
			ApplyInBetweenPatternV2<
				OrPatternV2<StarredExpressionPattern,
					GroupPatternsV2<NamedExpressionPattern,
						NegativeLookAheadV2<SingleTokenPatternV2<Token::TokenType::EQUAL>>>>,
				SingleTokenPatternV2<Token::TokenType::COMMA>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>, KwargsPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("','.(starred_expression | named_expression !'=')+ [',' kwargs ]");
			auto [starred_expressions_or_named_expressions, kwargs_] = *result;
			std::vector<std::shared_ptr<ASTNode>> args;
			args.reserve(starred_expressions_or_named_expressions.size());
			std::transform(starred_expressions_or_named_expressions.begin(),
				starred_expressions_or_named_expressions.end(),
				std::back_inserter(args),
				[](const auto &el) -> std::shared_ptr<ASTNode> {
					if (std::holds_alternative<std::shared_ptr<Starred>>(el)) {
						return std::get<std::shared_ptr<Starred>>(el);
					} else {
						auto el_ =
							std::get<std::tuple<std::shared_ptr<ASTNode>, std::monostate>>(el);
						auto [node, _] = el_;
						return node;
					}
				});

			if (kwargs_.has_value()) {
				auto [_, kwargs] = *kwargs_;
				return { { args, kwargs } };
			}
			return { { args, {} } };
		}

		using pattern2 = PatternMatchV2<KwargsPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("kwargs");
			auto [kwargs] = *result;
			return { { {}, kwargs } };
		}
		return {};
	}
};


struct ArgumentsPattern : PatternV2<ArgumentsPattern>
{
	using ResultType = typename traits<ArgumentsPattern>::result_type;

	// arguments:
	//     | args [','] &')'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("arguments");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		using pattern1 = PatternMatchV2<ArgsPattern,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>,
			LookAheadV2<SingleTokenPatternV2<Token::TokenType::RPAREN>>>;
		if (auto result = pattern1::match(p)) {
			auto [args, c, _] = *result;
			(void)c;
			return args;
		}
		return {};
	}
};

template<> struct traits<struct SlicePattern>
{
	using result_type = std::variant<Subscript::Index, Subscript::Slice>;
};

struct SlicePattern : PatternV2<SlicePattern>
{
	using ResultType = typename traits<SlicePattern>::result_type;

	// slice:
	//     | [expression] ':' [expression] [':' [expression] ]
	//     | expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("SlicePattern");

		using pattern1 = PatternMatchV2<ZeroOrOnePatternV2<ExpressionPattern>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			ZeroOrOnePatternV2<ExpressionPattern>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COLON>,
				ZeroOrOnePatternV2<ExpressionPattern>>>;
		if (auto result = pattern1::match(p)) {
			auto [lower, colon_token, upper, step_] = *result;

			Subscript::Slice slice{ lower.value_or(nullptr), upper.value_or(nullptr), nullptr };

			if (step_.has_value()) {
				auto [_, step] = *step_;
				slice.step = step.value_or(nullptr);
			}

			return slice;
		}

		// | expression
		using pattern2 = PatternMatchV2<ExpressionPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("expression");
			auto [value] = *result;
			return Subscript::Index{ value };
		}

		return {};
	}
};

struct SlicesPattern : PatternV2<SlicesPattern>
{
	using ResultType = typename traits<SlicesPattern>::result_type;

	// slices:
	//     | slice !','
	//     | ','.slice+ [',']
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("SlicesPattern");

		// slice !','
		using pattern1 = PatternMatchV2<SlicePattern,
			NegativeLookAheadV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("slice !','");
			auto [slice, _] = *result;
			if (std::holds_alternative<Subscript::Index>(slice)) {
				return std::get<Subscript::Index>(slice);
			}
			return std::get<Subscript::Slice>(slice);
		}

		// ','.slice+ [',']
		using pattern2 = PatternMatchV2<
			ApplyInBetweenPatternV2<SlicePattern, SingleTokenPatternV2<Token::TokenType::COMMA>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("','.slice+ [',']");
			auto [slices, _] = *result;
			Subscript::ExtSlice dims{ slices };
			return dims;
		}

		return {};
	}
};

template<> struct traits<struct PrimaryPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct PrimaryPattern : PatternV2<PrimaryPattern>
{
	using ResultType = typename traits<PrimaryPattern>::result_type;

	// primary:
	//     | invalid_primary  # must be before 'primary genexp' because of invalid_genexp
	//     | primary '.' NAME
	//     | primary genexp
	//     | primary '(' [arguments] ')'
	//     | primary '[' slices ']'
	//     | atom
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		//  primary '.' NAME
		DEBUG_LOG("PrimaryPattern");
		using pattern2 = PatternMatchV2<PrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::DOT>,
			NAMEPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG(" primary '.' NAME");
			auto [value, _, name_token] = *result;
			std::string name{ name_token.token.start().pointer_to_program,
				name_token.token.end().pointer_to_program };
			return std::make_shared<Attribute>(value,
				name,
				ContextType::LOAD,
				SourceLocation{ value->source_location().start, name_token.token.end() });
		}

		//  primary genexp
		using pattern3 = PatternMatchV2<PrimaryPattern, GenexPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("primary genexp");
			auto [function, arg] = *result;
			std::vector<std::shared_ptr<ASTNode>> args{ arg };
			std::vector<std::shared_ptr<Keyword>> kwargs;
			return std::make_shared<Call>(function,
				args,
				kwargs,
				SourceLocation{ function->source_location().start, arg->source_location().end });
		}

		// primary '(' [arguments] ')'
		using pattern4 = PatternMatchV2<PrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::LPAREN>,
			ZeroOrOnePatternV2<ArgumentsPattern>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("primary '(' [arguments] ')'");
			std::vector<std::shared_ptr<ASTNode>> args;
			std::vector<std::shared_ptr<Keyword>> kwargs;
			auto [function, _, arguments, r] = *result;
			if (arguments.has_value()) {
				auto [args_, kwargs_] = *arguments;
				args = std::move(args_);
				kwargs = std::move(kwargs_);
			}
			return std::make_shared<Call>(function,
				args,
				kwargs,
				SourceLocation{ function->source_location().start, r.token.end() });
		}

		// primary '[' slices ']'
		using pattern5 = PatternMatchV2<PrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPatternV2<Token::TokenType::RSQB>>;
		if (auto result = pattern5::match(p)) {
			DEBUG_LOG("'[' slices ']'");
			auto [value, l, slices, r] = *result;
			return std::make_shared<Subscript>(value,
				slices,
				ContextType::LOAD,
				SourceLocation{ value->source_location().start, r.token.end() });
		}

		using pattern6 = PatternMatchV2<AtomPattern>;
		if (auto result = pattern6::match(p)) {
			auto [atom] = *result;
			return atom;
		}
		return {};
	}
};

template<> struct traits<struct AwaitPrimaryPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct AwaitPrimaryPattern : PatternV2<AwaitPrimaryPattern>
{
	using ResultType = typename traits<AwaitPrimaryPattern>::result_type;

	// await_primary:
	//     | AWAIT primary
	//     | primary
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("AwaitPrimaryPattern");

		// AWAIT primary
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, AwaitKeywordPattern>,
			PrimaryPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("AWAIT primary");
			auto [await_token, el] = *result;
			SourceLocation sc{ await_token.start(), el->source_location().end };
			return std::make_shared<Await>(std::move(el), std::move(sc));
		}

		// primary
		using pattern2 = PatternMatchV2<PrimaryPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("primary");
			auto [el] = *result;
			return el;
		}

		return {};
	}
};

template<> struct traits<struct FactorPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct PowerPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct PowerPattern : PatternV2<PowerPattern>
{
	using ResultType = typename traits<PowerPattern>::result_type;

	// power:
	//     | await_primary '**' factor
	//     | await_primary
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("PowerPattern");
		// await_primary '**' factor
		using pattern1 = PatternMatchV2<AwaitPrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::DOUBLESTAR>,
			FactorPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("await_primary '**' factor");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::EXP,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		// await_primary
		using pattern2 = PatternMatchV2<AwaitPrimaryPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("await_primary");
			auto [el] = *result;
			return el;
		}

		return {};
	}
};

struct FactorPattern : PatternV2<FactorPattern>
{
	using ResultType = typename traits<FactorPattern>::result_type;

	// factor:
	//     | '+' factor
	//     | '-' factor
	//     | '~' factor
	//     | power
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("FactorPattern");
		// '+' factor
		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::PLUS>, FactorPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'+' factor");
			auto [start_token, el] = *result;
			return std::make_shared<UnaryExpr>(UnaryOpType::ADD,
				el,
				SourceLocation{ start_token.token.start(), el->source_location().end });
		}

		// '-' factor
		using pattern2 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::MINUS>, FactorPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'-' factor");
			auto [start_token, el] = *result;
			return std::make_shared<UnaryExpr>(UnaryOpType::SUB,
				el,
				SourceLocation{ start_token.token.start(), el->source_location().end });
		}

		// '~' factor
		using pattern3 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::TILDE>, FactorPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("'~' factor");
			auto [start_token, el] = *result;
			return std::make_shared<UnaryExpr>(UnaryOpType::INVERT,
				el,
				SourceLocation{ start_token.token.start(), el->source_location().end });
		}

		// power
		using pattern4 = PatternMatchV2<PowerPattern>;
		if (auto result = pattern4::match(p)) {
			auto [el] = *result;
			return el;
		}

		return {};
	}
};

template<> struct traits<struct TermPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct TermPattern : PatternV2<TermPattern>
{
	using ResultType = typename traits<TermPattern>::result_type;

	// term:
	//     | term '*' factor
	//     | term '/' factor
	//     | term '//' factor
	//     | term '%' factor
	//     | term '@' factor
	//     | factor
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("TermPattern");

		using pattern1 = PatternMatchV2<TermPattern,
			SingleTokenPatternV2<Token::TokenType::STAR>,
			FactorPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("term '*' factor");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::MULTIPLY,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		using pattern2 = PatternMatchV2<TermPattern,
			SingleTokenPatternV2<Token::TokenType::SLASH>,
			FactorPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("term '/' factor");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::SLASH,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		using pattern3 = PatternMatchV2<TermPattern,
			SingleTokenPatternV2<Token::TokenType::DOUBLESLASH>,
			FactorPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("term '//' factor");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::FLOORDIV,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		using pattern4 = PatternMatchV2<TermPattern,
			SingleTokenPatternV2<Token::TokenType::PERCENT>,
			FactorPattern>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("term '%' factor");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::MODULO,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		using pattern5 =
			PatternMatchV2<TermPattern, SingleTokenPatternV2<Token::TokenType::AT>, FactorPattern>;
		if (auto result = pattern5::match(p)) {
			DEBUG_LOG("term '@' factor");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::MATMUL,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		// factor
		using pattern6 = PatternMatchV2<FactorPattern>;
		if (auto result = pattern6::match(p)) {
			auto [factor] = *result;
			return factor;
		}

		return {};
	}
};

template<> struct traits<struct SumPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct SumPattern : PatternV2<SumPattern>
{
	using ResultType = typename traits<SumPattern>::result_type;
	// left recursive
	// sum:
	//     | sum '+' term
	//     | sum '-' term
	//     | term

	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("SumPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		// sum '+' term
		using pattern1 =
			PatternMatchV2<SumPattern, SingleTokenPatternV2<Token::TokenType::PLUS>, TermPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("sum '+' term");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::PLUS,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		// sum '-' term
		using pattern2 =
			PatternMatchV2<SumPattern, SingleTokenPatternV2<Token::TokenType::MINUS>, TermPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("sum '-' term");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::MINUS,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		// term
		using pattern3 = PatternMatchV2<TermPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("term");
			auto [term] = *result;
			return term;
		}

		return {};
	}
};


template<> struct traits<struct ShiftExprPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct ShiftExprPattern : PatternV2<ShiftExprPattern>
{
	using ResultType = typename traits<ShiftExprPattern>::result_type;

	// shift_expr:
	//     | shift_expr '<<' sum
	//     | shift_expr '>>' sum
	//     | sum
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("ShiftExprPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		using pattern1 = PatternMatchV2<ShiftExprPattern,
			SingleTokenPatternV2<Token::TokenType::LEFTSHIFT>,
			SumPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("shift_expr '<<' sum");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::LEFTSHIFT,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		using pattern2 = PatternMatchV2<ShiftExprPattern,
			SingleTokenPatternV2<Token::TokenType::RIGHTSHIFT>,
			SumPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("shift_expr '>>' sum");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::RIGHTSHIFT,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		using pattern3 = PatternMatchV2<SumPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("sum");
			auto [sum] = *result;
			return sum;
		}
		return {};
	}
};

template<> struct traits<struct BitwiseAndPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct BitwiseAndPattern : PatternV2<BitwiseAndPattern>
{
	using ResultType = typename traits<BitwiseAndPattern>::result_type;
	// bitwise_and:
	//     | bitwise_and '&' shift_expr
	//     | shift_expr
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("bitwise_and");

		// bitwise_and '&' shift_expr
		using pattern1 = PatternMatchV2<BitwiseAndPattern,
			SingleTokenPatternV2<Token::TokenType::AMPER>,
			ShiftExprPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("bitwise_and '&' shift_expr");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::AND,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		using pattern2 = PatternMatchV2<ShiftExprPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("shift_expr");
			auto [shift_expr] = *result;
			return shift_expr;
		}

		return {};
	}
};

template<> struct traits<struct BitwiseXorPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct BitwiseXorPattern : PatternV2<BitwiseXorPattern>
{
	using ResultType = typename traits<BitwiseXorPattern>::result_type;

	// bitwise_xor:
	//     | bitwise_xor '^' bitwise_and
	//     | bitwise_and
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("BitwiseXorPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		// bitwise_xor '^' bitwise_and
		using pattern1 = PatternMatchV2<BitwiseXorPattern,
			SingleTokenPatternV2<Token::TokenType::CIRCUMFLEX>,
			BitwiseAndPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("bitwise_xor '^' bitwise_and");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::XOR,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		// bitwise_and
		using pattern2 = PatternMatchV2<BitwiseAndPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("bitwise_and");
			auto [and_op] = *result;
			return and_op;
		}

		return {};
	}
};


struct BitwiseOrPattern : PatternV2<BitwiseOrPattern>
{
	using ResultType = typename traits<BitwiseOrPattern>::result_type;

	// bitwise_or:
	//     | bitwise_or '|' bitwise_xor
	//     | bitwise_xor
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("BitwiseOrPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		// bitwise_or '|' bitwise_xor
		using pattern1 = PatternMatchV2<BitwiseOrPattern,
			SingleTokenPatternV2<Token::TokenType::VBAR>,
			BitwiseXorPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("bitwise_or '|' bitwise_xor");
			auto [lhs, _, rhs] = *result;
			return std::make_shared<BinaryExpr>(BinaryOpType::OR,
				lhs,
				rhs,
				SourceLocation{ lhs->source_location().start, rhs->source_location().end });
		}

		// bitwise_xor
		using pattern2 = PatternMatchV2<BitwiseXorPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("bitwise_xor");
			auto [bitwise_xor] = *result;
			return bitwise_xor;
		}

		return {};
	}
};

template<> struct traits<struct EqBitwiseOrPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct EqBitwiseOrPattern : PatternV2<EqBitwiseOrPattern>
{
	using ResultType = typename traits<EqBitwiseOrPattern>::result_type;

	// eq_bitwise_or: '==' bitwise_or
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("eq_bitwise_or");

		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::EQEQUAL>, BitwiseOrPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'==' bitwise_or");
			auto [_, rhs] = *result;
			return rhs;
		}
		return {};
	}
};

template<> struct traits<struct NotEqBitwiseOrPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct NotEqBitwiseOrPattern : PatternV2<NotEqBitwiseOrPattern>
{
	using ResultType = typename traits<NotEqBitwiseOrPattern>::result_type;

	// noteq_bitwise_or: '!=' bitwise_or
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("noteq_bitwise_or");
		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::NOTEQUAL>, BitwiseOrPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'!=' bitwise_or");
			auto [_, rhs] = *result;
			return rhs;
		}
		return {};
	}
};

template<> struct traits<struct LtEqBitwiseOrPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct LtEqBitwiseOrPattern : PatternV2<LtEqBitwiseOrPattern>
{
	using ResultType = typename traits<LtEqBitwiseOrPattern>::result_type;

	// lteq_bitwise_or: '<=' bitwise_or
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LESSEQUAL>, BitwiseOrPattern>;
		DEBUG_LOG("LtEqBitwiseOrPattern");
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'<=' bitwise_or");
			auto [_, rhs] = *result;
			return rhs;
		}
		return {};
	}
};

template<> struct traits<struct LtBitwiseOrPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct LtBitwiseOrPattern : PatternV2<LtBitwiseOrPattern>
{
	using ResultType = typename traits<LtBitwiseOrPattern>::result_type;

	// lteq_bitwise_or: '<' bitwise_or
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("LtBitwiseOrPattern");
		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LESS>, BitwiseOrPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'<' bitwise_or");
			auto [_, rhs] = *result;
			return rhs;
		}
		return {};
	}
};

template<> struct traits<struct GtEqBitwiseOrPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct GtEqBitwiseOrPattern : PatternV2<GtEqBitwiseOrPattern>
{
	using ResultType = typename traits<GtEqBitwiseOrPattern>::result_type;

	// lteq_bitwise_or: '<' bitwise_or
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::GREATEREQUAL>, BitwiseOrPattern>;
		DEBUG_LOG("GtEqBitwiseOrPattern");
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'>=' bitwise_or");
			auto [_, rhs] = *result;
			return rhs;
		}
		return {};
	}
};

template<> struct traits<struct GtBitwiseOrPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct GtBitwiseOrPattern : PatternV2<GtBitwiseOrPattern>
{
	using ResultType = typename traits<GtBitwiseOrPattern>::result_type;

	// lteq_bitwise_or: '<' bitwise_or
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::GREATER>, BitwiseOrPattern>;
		DEBUG_LOG("GtBitwiseOrPattern");
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'>' bitwise_or");
			auto [_, rhs] = *result;
			return rhs;
		}
		return {};
	}
};

template<> struct traits<struct InBitwiseOrPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct InBitwiseOrPattern : PatternV2<InBitwiseOrPattern>
{
	using ResultType = typename traits<InBitwiseOrPattern>::result_type;

	// in_bitwise_or: 'in' bitwise_or
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, InKeywordPattern>,
			BitwiseOrPattern>;
		DEBUG_LOG("InBitwiseOrPattern");
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'in' bitwise_or ");
			auto [_, rhs] = *result;
			return rhs;
		}
		return {};
	}
};

template<> struct traits<struct NotInBitwiseOrPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct NotInBitwiseOrPattern : PatternV2<NotInBitwiseOrPattern>
{
	using ResultType = typename traits<NotInBitwiseOrPattern>::result_type;

	// notin_bitwise_or: 'not' 'in' bitwise_or
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, NotKeywordPattern>,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, InKeywordPattern>,
			BitwiseOrPattern>;
		DEBUG_LOG("NotInBitwiseOrPattern");
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'not' 'in' bitwise_or ");
			auto [not_token, in_token, rhs] = *result;
			return rhs;
		}
		return {};
	}
};

template<> struct traits<struct IsNotBitwiseOrPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct IsNotBitwiseOrPattern : PatternV2<IsNotBitwiseOrPattern>
{
	using ResultType = typename traits<IsNotBitwiseOrPattern>::result_type;

	// is_bitwise_or: 'is' 'not' bitwise_or
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, IsKeywordPattern>,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, NotKeywordPattern>,
			BitwiseOrPattern>;
		DEBUG_LOG("IsNotBitwiseOrPattern");
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'is not' bitwise_or ");
			auto [is_token, not_token, rhs] = *result;
			return rhs;
		}
		return {};
	}
};

template<> struct traits<struct IsBitwiseOrPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct IsBitwiseOrPattern : PatternV2<IsBitwiseOrPattern>
{
	using ResultType = typename traits<IsBitwiseOrPattern>::result_type;

	// is_bitwise_or: 'is' bitwise_or
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, IsKeywordPattern>,
			BitwiseOrPattern>;
		DEBUG_LOG("IsBitwiseOrPattern");
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'is' bitwise_or ");
			auto [_, rhs] = *result;
			return rhs;
		}
		return {};
	}
};

template<> struct traits<struct CompareOpBitwiseOrPairPattern>
{
	using result_type = std::pair<std::shared_ptr<ASTNode>, Compare::OpType>;
};

struct CompareOpBitwiseOrPairPattern : PatternV2<CompareOpBitwiseOrPairPattern>
{
	using ResultType = typename traits<CompareOpBitwiseOrPairPattern>::result_type;

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
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("compare_op_bitwise_or_pair");
		// eq_bitwise_or
		using pattern1 = PatternMatchV2<EqBitwiseOrPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("eq_bitwise_or");
			auto [eq_bitwise_or] = *result;
			return { { eq_bitwise_or, Compare::OpType::Eq } };
		}
		// noteq_bitwise_or
		using pattern2 = PatternMatchV2<NotEqBitwiseOrPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("not_eq_bitwise_or");
			auto [not_eq_bitwise_or] = *result;
			return { { not_eq_bitwise_or, Compare::OpType::NotEq } };
		}
		// lte_bitwise_or
		using pattern3 = PatternMatchV2<LtEqBitwiseOrPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("lte_bitwise_or");
			auto [lte_bitwise_or] = *result;
			return { { lte_bitwise_or, Compare::OpType::LtE } };
		}
		// lt_bitwise_or
		using pattern4 = PatternMatchV2<LtBitwiseOrPattern>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("lt_bitwise_or");
			auto [lt_bitwise_or] = *result;
			return { { lt_bitwise_or, Compare::OpType::Lt } };
		}
		// gte_bitwise_or
		using pattern5 = PatternMatchV2<GtEqBitwiseOrPattern>;
		if (auto result = pattern5::match(p)) {
			DEBUG_LOG("gte_bitwise_or");
			auto [gte_bitwise_or] = *result;
			return { { gte_bitwise_or, Compare::OpType::GtE } };
		}
		// gt_bitwise_or
		using pattern6 = PatternMatchV2<GtBitwiseOrPattern>;
		if (auto result = pattern6::match(p)) {
			DEBUG_LOG("gt_bitwise_or");
			auto [gt_bitwise_or] = *result;
			return { { gt_bitwise_or, Compare::OpType::Gt } };
		}
		// notin_bitwise_or
		using pattern7 = PatternMatchV2<NotInBitwiseOrPattern>;
		if (auto result = pattern7::match(p)) {
			DEBUG_LOG("notin_bitwise_or");
			auto [notin_bitwise_or] = *result;
			return { { notin_bitwise_or, Compare::OpType::NotIn } };
		}
		// in_bitwise_or
		using pattern8 = PatternMatchV2<InBitwiseOrPattern>;
		if (auto result = pattern8::match(p)) {
			DEBUG_LOG("in_bitwise_or");
			auto [in_bitwise_or] = *result;
			return { { in_bitwise_or, Compare::OpType::In } };
		}
		// isnot_bitwise_or
		using pattern9 = PatternMatchV2<IsNotBitwiseOrPattern>;
		if (auto result = pattern9::match(p)) {
			DEBUG_LOG("isnot_bitwise_or");
			auto [isnot_bitwise_or] = *result;
			return { { isnot_bitwise_or, Compare::OpType::IsNot } };
		}
		// is_bitwise_or
		using pattern10 = PatternMatchV2<IsBitwiseOrPattern>;
		if (auto result = pattern10::match(p)) {
			DEBUG_LOG("is_bitwise_or");
			auto [is_bitwise_or] = *result;
			return { { is_bitwise_or, Compare::OpType::Is } };
		}
		return {};
	}
};

template<> struct traits<struct ComparissonPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct ComparissonPattern : PatternV2<ComparissonPattern>
{
	using ResultType = typename traits<ComparissonPattern>::result_type;

	// comparison:
	//     | bitwise_or compare_op_bitwise_or_pair+
	//     | bitwise_or
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("comparison");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		// bitwise_or compare_op_bitwise_or_pair+
		using pattern1 =
			PatternMatchV2<BitwiseOrPattern, OneOrMorePatternV2<CompareOpBitwiseOrPairPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("bitwise_or compare_op_bitwise_or_pair+");
			auto [lhs, compare_ops] = *result;
			std::vector<Compare::OpType> ops;
			std::vector<std::shared_ptr<ASTNode>> comparators;
			ops.reserve(compare_ops.size());
			comparators.reserve(compare_ops.size());
			for (auto [comparator, op] : compare_ops) {
				ops.push_back(op);
				comparators.push_back(std::move(comparator));
			}
			return std::make_shared<Compare>(lhs,
				std::move(ops),
				std::move(comparators),
				SourceLocation{
					lhs->source_location().start, comparators.back()->source_location().end });
		}
		// bitwise_or
		using pattern2 = PatternMatchV2<BitwiseOrPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("bitwise_or");
			auto [bitwise_or] = *result;
			return bitwise_or;
		}

		return {};
	}
};

template<> struct traits<struct InversionPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct InversionPattern : PatternV2<InversionPattern>
{
	using ResultType = typename traits<InversionPattern>::result_type;

	// inversion:
	//     | 'not' inversion
	//     | comparison
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("InversionPattern");
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, NotKeywordPattern>,
			ComparissonPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'not' inversion");
			auto [not_token, inversion] = *result;
			return std::make_shared<UnaryExpr>(UnaryOpType::NOT,
				inversion,
				SourceLocation{ not_token.start(), inversion->source_location().end });
		}

		// comparison
		using pattern2 = PatternMatchV2<ComparissonPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("comparison");
			auto [comparison] = *result;
			return comparison;
		}

		return {};
	}
};

template<> struct traits<struct ConjunctionPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct ConjunctionPattern : PatternV2<ConjunctionPattern>
{
	using ResultType = typename traits<ConjunctionPattern>::result_type;

	// conjunction:
	//     | inversion ('and' inversion )+
	//     | inversion
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("ConjunctionPattern");

		// inversion ('and' inversion )+
		using pattern1 = PatternMatchV2<InversionPattern,
			OneOrMorePatternV2<
				AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, AndKeywordPattern>,
				InversionPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("inversion ('and' inversion )+");

			auto [inversion, inversions] = *result;
			std::vector<std::shared_ptr<ASTNode>> values;
			values.reserve(1 + inversions.size());
			values.push_back(inversion);
			std::transform(inversions.begin(),
				inversions.end(),
				std::back_inserter(values),
				[](const std::tuple<Token, std::shared_ptr<ast::ASTNode>> &el) {
					auto [token, val] = el;
					return val;
				});
			return std::make_shared<BoolOp>(BoolOp::OpType::And,
				values,
				SourceLocation{ values.front()->source_location().start,
					values.back()->source_location().end });
		}

		// inversion
		using pattern2 = PatternMatchV2<InversionPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("inversion");
			auto [inversion] = *result;
			return inversion;
		}

		return {};
	}
};

struct DisjunctionPattern : PatternV2<DisjunctionPattern>
{
	using ResultType = typename traits<DisjunctionPattern>::result_type;

	// disjunction:
	//     | conjunction ('or' conjunction )+
	//     | conjunction
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("DisjunctionPattern");
		// conjunction ('or' conjunction )+
		using pattern1 = PatternMatchV2<ConjunctionPattern,
			OneOrMorePatternV2<
				AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, OrKeywordPattern>,
				ConjunctionPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("conjunction ('or' conjunction )+");

			auto [conjunction, conjunctions] = *result;
			std::vector<std::shared_ptr<ASTNode>> values;
			values.reserve(1 + conjunctions.size());
			values.push_back(conjunction);
			std::transform(conjunctions.begin(),
				conjunctions.end(),
				std::back_inserter(values),
				[](const std::tuple<Token, std::shared_ptr<ast::ASTNode>> &el) {
					auto [token, val] = el;
					return val;
				});
			return std::make_shared<BoolOp>(BoolOp::OpType::Or,
				values,
				SourceLocation{ values.front()->source_location().start,
					values.back()->source_location().end });
		}

		// conjunction
		using pattern2 = PatternMatchV2<ConjunctionPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("conjunction");
			auto [conjunction] = *result;
			return conjunction;
		}

		return {};
	}
};

template<> struct traits<struct LambdaParamPattern>
{
	using result_type = Token;
};

struct LambdaParamPattern : PatternV2<LambdaParamPattern>
{
	using ResultType = typename traits<LambdaParamPattern>::result_type;

	// lambda_param: NAME
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("lambda_param");
		using pattern1 = PatternMatchV2<NAMEPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("NAME");
			auto [name] = *result;
			return name.token;
		}

		return {};
	}
};

template<> struct traits<struct LambdaParamNoDefaultPattern>
{
	using result_type = std::shared_ptr<Argument>;
};

struct LambdaParamNoDefaultPattern : PatternV2<LambdaParamNoDefaultPattern>
{
	using ResultType = typename traits<LambdaParamNoDefaultPattern>::result_type;

	// lambda_param_no_default:
	//     | lambda_param ','
	//     | lambda_param &':'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("lambda_param_no_default");

		// lambda_param ','
		using pattern1 =
			PatternMatchV2<LambdaParamPattern, SingleTokenPatternV2<Token::TokenType::COMMA>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("lambda_param ','");
			auto [lambda_param, comma_token] = *result;
			std::string parameter_name{
				lambda_param.start().pointer_to_program,
				lambda_param.end().pointer_to_program,
			};
			return std::make_shared<Argument>(parameter_name,
				nullptr,
				"",
				SourceLocation{ lambda_param.start(), comma_token.token.end() });
		}

		// lambda_param &':'
		using pattern2 = PatternMatchV2<LambdaParamPattern,
			LookAheadV2<SingleTokenPatternV2<Token::TokenType::COLON>>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("lambda_param &':'");
			auto [lambda_param, _] = *result;
			std::string parameter_name{
				lambda_param.start().pointer_to_program,
				lambda_param.end().pointer_to_program,
			};
			return std::make_shared<Argument>(parameter_name,
				nullptr,
				"",
				SourceLocation{ lambda_param.start(), lambda_param.end() });
		}

		return {};
	}
};

template<> struct traits<struct DefaultPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct LambdaParamWithDefaultPattern>
{
	using result_type = std::pair<std::shared_ptr<Argument>, std::shared_ptr<ASTNode>>;
};

struct LambdaParamWithDefaultPattern : PatternV2<LambdaParamWithDefaultPattern>
{
	using ResultType = typename traits<LambdaParamWithDefaultPattern>::result_type;

	// lambda_param_with_default:
	//     | lambda_param default ','
	//     | lambda_param default &':'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("lambda_param_with_default");

		// lambda_param default ','
		using pattern1 = PatternMatchV2<LambdaParamPattern,
			DefaultPattern,
			SingleTokenPatternV2<Token::TokenType::COMMA>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("lambda_param default ','");
			auto [lambda_param, default_, comma_token] = *result;
			std::string parameter_name{
				lambda_param.start().pointer_to_program,
				lambda_param.end().pointer_to_program,
			};
			return { {
				std::make_shared<Argument>(parameter_name,
					nullptr,
					"",
					SourceLocation{ lambda_param.start(), comma_token.token.end() }),
				default_,
			} };
		}

		// lambda_param default &':'
		using pattern2 = PatternMatchV2<LambdaParamPattern,
			DefaultPattern,
			LookAheadV2<SingleTokenPatternV2<Token::TokenType::COLON>>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("lambda_param default &':'");
			auto [lambda_param, default_, _] = *result;
			std::string parameter_name{
				lambda_param.start().pointer_to_program,
				lambda_param.end().pointer_to_program,
			};
			return { {
				std::make_shared<Argument>(parameter_name,
					nullptr,
					"",
					SourceLocation{ lambda_param.start(), default_->source_location().end }),
				default_,
			} };
		}

		return {};
	}
};

template<> struct traits<struct LambdaStarEtcPattern>
{
	using result_type = std::tuple<std::optional<std::shared_ptr<Argument>>,
		std::vector<std::pair<std::shared_ptr<Argument>, std::optional<std::shared_ptr<ASTNode>>>>,
		std::optional<std::shared_ptr<Argument>>>;
};

struct LambdaStarEtcPattern : PatternV2<LambdaStarEtcPattern>
{
	using ResultType = typename traits<LambdaStarEtcPattern>::result_type;

	// lambda_star_etc:
	//     | '*' lambda_param_no_default lambda_param_maybe_default* [lambda_kwds]
	//     | '*' ',' lambda_param_maybe_default+ [lambda_kwds]
	//     | lambda_kwds
	static std::optional<ResultType> matches_impl(Parser &) { return {}; }
};

template<> struct traits<struct LambdaParametersPattern>
{
	using result_type = std::shared_ptr<Arguments>;
};

struct LambdaParametersPattern : PatternV2<LambdaParametersPattern>
{
	using ResultType = typename traits<LambdaParametersPattern>::result_type;

	// lambda_parameters:
	//     | lambda_slash_no_default lambda_param_no_default* lambda_param_with_default*
	//     	 [lambda_star_etc]
	//     | lambda_slash_with_default lambda_param_with_default* [lambda_star_etc]
	//     | lambda_param_no_default+ lambda_param_with_default* [lambda_star_etc]
	//     | lambda_param_with_default+ [lambda_star_etc]
	//     | lambda_star_etc
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern3 = PatternMatchV2<OneOrMorePatternV2<LambdaParamNoDefaultPattern>,
			ZeroOrMorePatternV2<LambdaParamWithDefaultPattern>,
			ZeroOrOnePatternV2<LambdaStarEtcPattern>>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("lambda_param_no_default+ lambda_param_with_default* [lambda_star_etc]");
			auto [lambda_param_no_default, lambda_param_with_default, lambda_star_etc] = *result;

			std::vector<std::shared_ptr<Argument>> posonlyargs;
			std::vector<std::shared_ptr<Argument>> args;
			std::shared_ptr<Argument> vararg;
			std::vector<std::shared_ptr<Argument>> kwonlyargs;
			std::vector<std::shared_ptr<ASTNode>> kw_defaults;
			std::shared_ptr<Argument> kwarg;
			std::vector<std::shared_ptr<ASTNode>> defaults;
			SourceLocation source_location{
				lambda_param_no_default.front()->source_location().start,
				p.lexer().peek_token(p.token_position() - 1)->end(),// too lazy to figure this out
			};

			args.insert(args.end(), lambda_param_no_default.begin(), lambda_param_no_default.end());
			for (const auto &[param, default_] : lambda_param_with_default) {
				args.push_back(param);
				defaults.push_back(default_);
			}

			if (lambda_star_etc.has_value()) {
				auto [vararg_, kwonlyargs_, kwarg_] = *lambda_star_etc;
				vararg = vararg_.value_or(nullptr);
				for (const auto &[param, default_] : kwonlyargs_) {
					kwonlyargs.push_back(param);
					kw_defaults.push_back(default_.value_or(nullptr));
				}
				kwarg = kwarg_.value_or(nullptr);
			}

			return std::make_shared<Arguments>(posonlyargs,
				args,
				vararg,
				kwonlyargs,
				kw_defaults,
				kwarg,
				defaults,
				source_location);
		}
		return {};
	}
};

template<> struct traits<struct LambdaParamsPattern>
{
	using result_type = typename traits<LambdaParametersPattern>::result_type;
};

struct LambdaParamsPattern : PatternV2<LambdaParamsPattern>
{
	using ResultType = typename traits<LambdaParametersPattern>::result_type;

	// lambda_params: lambda_parameters
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<LambdaParametersPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("lambda_parameters");
			auto [lambda_parameters] = *result;
			return lambda_parameters;
		}

		return {};
	}
};

template<> struct traits<struct LambDefPattern>
{
	using result_type = std::shared_ptr<Lambda>;
};

struct LambDefPattern : PatternV2<LambDefPattern>
{
	using ResultType = typename traits<LambDefPattern>::result_type;

	// lambdef: 'lambda' [lambda_params] ':' expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, LambdaKeywordPattern>,
			ZeroOrOnePatternV2<LambdaParamsPattern>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			ExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'lambda' [lambda_params] ':' expression");
			auto [lambda_token, params, token, body] = *result;
			return std::make_shared<Lambda>(
				params.value_or(
					std::make_shared<Arguments>(std::vector<std::shared_ptr<Argument>>{},
						SourceLocation{ lambda_token.end(), body->source_location().start })),
				body,
				SourceLocation{ lambda_token.start(), body->source_location().end });
		}

		return {};
	}
};

struct ExpressionPattern : PatternV2<ExpressionPattern>
{
	using ResultType = typename traits<ExpressionPattern>::result_type;
	// expression:
	//     | disjunction 'if' disjunction 'else' expression
	//     | disjunction
	//     | lambdef
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("ExpressionPattern");
		DEBUG_LOG("{}", p.lexer().peek_token(p.token_position())->to_string());
		using pattern1 = PatternMatchV2<DisjunctionPattern,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, IfKeywordPattern>,
			DisjunctionPattern,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ElseKeywordPattern>,
			ExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("disjunction 'if' disjunction 'else' expression");
			auto [body, if_token, test, else_token, orelse] = *result;
			return std::make_shared<IfExpr>(test,
				body,
				orelse,
				SourceLocation{ test->source_location().start, orelse->source_location().end });
		}

		// disjunction
		using pattern2 = PatternMatchV2<DisjunctionPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("disjunction");
			auto [disjunction] = *result;
			return disjunction;
		}

		// lambdef
		using pattern3 = PatternMatchV2<LambDefPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("lambdef");
			auto [lambdef] = *result;
			return lambdef;
		}

		return {};
	}
};

template<> struct traits<struct StarExpressionPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct StarExpressionPattern : PatternV2<StarExpressionPattern>
{
	using ResultType = typename traits<StarExpressionPattern>::result_type;

	// star_expression:
	//     | '*' bitwise_or
	//     | expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// expression
		using pattern2 = PatternMatchV2<ExpressionPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("expression");
			auto [expression] = *result;
			return expression;
		}
		return {};
	}
};


struct StarExpressionsPattern : PatternV2<StarExpressionsPattern>
{
	using ResultType = typename traits<StarExpressionsPattern>::result_type;

	// star_expressions:
	// 	| star_expression (',' star_expression )+ [',']
	// 	| star_expression ','
	// 	| star_expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("star_expressions");

		// star_expression (',' star_expression )+ [',']
		using pattern1 = PatternMatchV2<StarExpressionPattern,
			OneOrMorePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>,
				StarExpressionPattern>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("star_expression (',' star_expression )+ [',']");
			auto [expression, more_expressions, _] = *result;
			std::vector<std::shared_ptr<ASTNode>> expressions;
			expressions.reserve(1 + more_expressions.size());
			expressions.push_back(expression);
			for (const auto &[_, expr] : more_expressions) { expressions.push_back(expr); }
			auto end_token = p.lexer().peek_token(p.token_position() - 1);
			return std::make_shared<Tuple>(expressions,
				ContextType::LOAD,
				SourceLocation{ expression->source_location().start, end_token->end() });
		}

		// star_expression ','
		using pattern2 =
			PatternMatchV2<StarExpressionPattern, SingleTokenPatternV2<Token::TokenType::COMMA>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("star_expression ','");
			auto [expression, token] = *result;
			return std::make_shared<Tuple>(std::vector{ expression },
				ContextType::LOAD,
				SourceLocation{ expression->source_location().start, token.token.end() });
		}

		// star_expression
		using pattern3 = PatternMatchV2<StarExpressionPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("star_expression");
			auto [expression] = *result;
			return expression;
		}
		return {};
	}
};

template<> struct traits<struct SingleSubscriptAttributeTargetPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct SingleSubscriptAttributeTargetPattern : PatternV2<SingleSubscriptAttributeTargetPattern>
{
	using ResultType = typename traits<SingleSubscriptAttributeTargetPattern>::result_type;

	// single_subscript_attribute_target:
	//     | t_primary '.' NAME !t_lookahead
	//     | t_primary '[' slices ']' !t_lookahead
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("single_subscript_attribute_target");

		// t_primary '.' NAME !t_lookahead
		using pattern1 = PatternMatchV2<TPrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::DOT>,
			NAMEPattern,
			NegativeLookAheadV2<TLookahead>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("t_primary '.' NAME !t_lookahead");
			auto [target, _, name_token, l] = *result;
			std::string name{ name_token.token.start().pointer_to_program,
				name_token.token.end().pointer_to_program };
			return std::make_shared<Attribute>(target,
				name,
				ContextType::STORE,
				SourceLocation{ target->source_location().start, name_token.token.end() });
		}

		// t_primary '[' slices ']' !t_lookahead
		using pattern2 = PatternMatchV2<TPrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPatternV2<Token::TokenType::RSQB>,
			NegativeLookAheadV2<TLookahead>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("t_primary '[' slices ']' !t_lookahead");
			auto [target, l, slices, r, _] = *result;
			return std::make_shared<Subscript>(target,
				slices,
				ContextType::STORE,
				SourceLocation{ target->source_location().start, r.token.end() });
			TODO();
		}

		return {};
	}
};

template<> struct traits<struct SingleTargetPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct SingleTargetPattern : PatternV2<SingleTargetPattern>
{
	using ResultType = typename traits<SingleTargetPattern>::result_type;

	// single_target:
	//     | single_subscript_attribute_target
	//     | NAME
	//     | '(' single_target ')'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("SingleTargetPattern");

		// single_subscript_attribute_target
		using pattern1 = PatternMatchV2<SingleSubscriptAttributeTargetPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("single_subscript_attribute_target");
			auto [target] = *result;
			return target;
		}

		// NAME
		using pattern2 = PatternMatchV2<NAMEPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("NAME");
			auto [name_token] = *result;
			std::string name{ name_token.token.start().pointer_to_program,
				name_token.token.end().pointer_to_program };
			return std::make_shared<Name>(name,
				ContextType::STORE,
				SourceLocation{ name_token.token.start(), name_token.token.end() });
		}

		using pattern3 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LPAREN>,
			SingleTargetPattern,
			SingleTokenPatternV2<Token::TokenType::RPAREN>>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("'(' single_target ')'");
			auto [l, target, r] = *result;
			return target;
		}

		return {};
	}
};

template<> struct traits<struct AugAssignPattern>
{
	using result_type = Token;
};

struct AugAssignPattern : PatternV2<AugAssignPattern>
{
	using ResultType = typename traits<AugAssignPattern>::result_type;

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
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("augassign");

		// '+='
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::PLUSEQUAL>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'+='");
			auto [token] = *result;
			return token.token;
		}

		// '-='
		using pattern2 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::MINEQUAL>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'-='");
			auto [token] = *result;
			return token.token;
		}

		// '*='
		using pattern3 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::STAREQUAL>>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("'*='");
			auto [token] = *result;
			return token.token;
		}

		// '@='
		using pattern4 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::ATEQUAL>>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("'@='");
			auto [token] = *result;
			return token.token;
		}

		// '/='
		using pattern5 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::SLASHEQUAL>>;
		if (auto result = pattern5::match(p)) {
			DEBUG_LOG("'/='");
			auto [token] = *result;
			return token.token;
		}

		// '%='
		using pattern6 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::PERCENTEQUAL>>;
		if (auto result = pattern6::match(p)) {
			DEBUG_LOG("'%='");
			auto [token] = *result;
			return token.token;
		}

		// '&='
		using pattern7 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::AMPEREQUAL>>;
		if (auto result = pattern7::match(p)) {
			DEBUG_LOG("'&='");
			auto [token] = *result;
			return token.token;
		}

		// '|='
		using pattern8 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::VBAREQUAL>>;
		if (auto result = pattern8::match(p)) {
			DEBUG_LOG("'|='");
			auto [token] = *result;
			return token.token;
		}

		// '^='
		using pattern9 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::CIRCUMFLEXEQUAL>>;
		if (auto result = pattern9::match(p)) {
			DEBUG_LOG("'^='");
			auto [token] = *result;
			return token.token;
		}

		// '<<='
		using pattern10 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LEFTSHIFTEQUAL>>;
		if (auto result = pattern10::match(p)) {
			DEBUG_LOG("'<<='");
			auto [token] = *result;
			return token.token;
		}

		// '>>='
		using pattern11 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::RIGHTSHIFTEQUAL>>;
		if (auto result = pattern11::match(p)) {
			DEBUG_LOG("'>>='");
			auto [token] = *result;
			return token.token;
		}

		//     | '**='
		using pattern12 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::DOUBLESTAREQUAL>>;
		if (auto result = pattern12::match(p)) {
			DEBUG_LOG("'**='");
			auto [token] = *result;
			return token.token;
		}

		//     | '//='
		using pattern13 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::DOUBLESLASHEQUAL>>;
		if (auto result = pattern13::match(p)) {
			DEBUG_LOG("'//='");
			auto [token] = *result;
			return token.token;
		}

		return {};
	}
};

struct YieldExpressionPattern : PatternV2<YieldExpressionPattern>
{
	using ResultType = typename traits<YieldExpressionPattern>::result_type;

	// yield_expr:
	// 		| 'yield' 'from' expression
	// 		| 'yield' [star_expressions]
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// 'yield' 'from' expression
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, YieldKeywordPattern>,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, FromKeywordPattern>,
			ExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'yield' 'from' expression");
			auto [yield_token, from_token, expression] = *result;
			return std::make_shared<YieldFrom>(expression,
				SourceLocation{ yield_token.start(), expression->source_location().end });
		}

		// 'yield' [star_expressions]
		using pattern2 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, YieldKeywordPattern>,
			ZeroOrOnePatternV2<StarExpressionsPattern>>;
		if (auto result = pattern2::match(p)) {
			auto [yield_token, star_expressions] = *result;
			if (star_expressions.has_value()) {
				return std::make_shared<Yield>(*star_expressions,
					SourceLocation{
						yield_token.start(), (*star_expressions)->source_location().end });
			}
			const auto end = p.lexer().peek_token(p.token_position() - 1);
			auto value = std::make_shared<Constant>(
				NameConstant{ NoneType{} }, SourceLocation{ end->start(), end->end() });
			return std::make_shared<Yield>(
				value, SourceLocation{ yield_token.start(), value->source_location().end });
		}

		return {};
	}
};

template<> struct traits<struct AnnotatedRhsPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct AnnotatedRhsPattern : PatternV2<AnnotatedRhsPattern>
{
	using ResultType = typename traits<AnnotatedRhsPattern>::result_type;

	// annotated_rhs: yield_expr | star_expressions
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("annotated_rhs");

		// yield_expr
		using pattern1 = PatternMatchV2<YieldExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("yield_expr");
			auto [yield_expr] = *result;
			return yield_expr;
		}

		// star_expressions
		using pattern2 = PatternMatchV2<YieldExpressionPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("star_expressions");
			auto [star_expressions] = *result;
			return star_expressions;
		}

		return {};
	}
};

template<> struct traits<struct AssignmentPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct AssignmentPattern : PatternV2<AssignmentPattern>
{
	using ResultType = typename traits<AssignmentPattern>::result_type;

	// assignment:
	// 	| NAME ':' expression ['=' annotated_rhs ]
	// 	| ('(' single_target ')'
	// 		| single_subscript_attribute_target) ':' expression ['=' annotated_rhs ]
	// 	| (star_targets '=' )+ (yield_expr | star_expressions) !'=' [TYPE_COMMENT]
	// 	| single_target augassign ~ (yield_expr | star_expressions)
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("assignment");

		// NAME ':' expression ['=' annotated_rhs ]
		using pattern1 = PatternMatchV2<NAMEPattern,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			ExpressionPattern,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::EQUAL>, AnnotatedRhsPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("NAME ':' expression ['=' annotated_rhs ]");
			auto [name_token, c, expression, _] = *result;
			TODO_NO_FAIL();
		}

		// (star_targets '=' )+ (yield_expr | star_expressions) !'=' [TYPE_COMMENT]
		using pattern3 = PatternMatchV2<
			OneOrMorePatternV2<StarTargetsPattern, SingleTokenPatternV2<Token::TokenType::EQUAL>>,
			OrPatternV2<YieldExpressionPattern, StarExpressionsPattern>>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("(star_targets '=' )+ (yield_expr | star_expressions) !'=' [TYPE_COMMENT]");
			auto [els, expressions] = *result;
			std::vector<std::shared_ptr<ASTNode>> target_elements;
			target_elements.reserve(els.size());
			for (const auto &[el, _] : els) { target_elements.push_back(el); }

			auto targets = std::make_shared<Tuple>(target_elements,
				ContextType::STORE,
				SourceLocation{ target_elements.front()->source_location().start,
					target_elements.back()->source_location().end });

			const auto start = targets->source_location().start;
			const auto end = expressions->source_location().end;

			if (targets->elements().size() == 1) {
				return std::make_shared<Assign>(
					std::vector<std::shared_ptr<ASTNode>>{ targets->elements().back() },
					expressions,
					"",
					SourceLocation{ start, end });
			} else {
				return std::make_shared<Assign>(
					targets->elements(), expressions, "", SourceLocation{ start, end });
			}
		}

		// single_target augassign ~ (yield_expr | star_expressions)
		using pattern4 = PatternMatchV2<SingleTargetPattern,
			AugAssignPattern,
			OrPatternV2<YieldExpressionPattern, StarExpressionsPattern>>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("single_target augassign ~ (yield_expr | star_expressions)");
			auto [target, assign, expression] = *result;
			SourceLocation source_location{ target->source_location().start,
				expression->source_location().end };
			switch (assign.token_type()) {
			case Token::TokenType::PLUSEQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::PLUS, expression, source_location);
			} break;
			case Token::TokenType::MINEQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::MINUS, expression, source_location);
			} break;
			case Token::TokenType::STAREQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::MULTIPLY, expression, source_location);
			} break;
			case Token::TokenType::ATEQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::MATMUL, expression, source_location);
			} break;
			case Token::TokenType::SLASHEQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::SLASH, expression, source_location);
			} break;
			case Token::TokenType::PERCENTEQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::MODULO, expression, source_location);
			} break;
			case Token::TokenType::AMPEREQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::AND, expression, source_location);
			} break;
			case Token::TokenType::VBAREQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::OR, expression, source_location);
			} break;
			case Token::TokenType::CIRCUMFLEXEQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::XOR, expression, source_location);
			} break;
			case Token::TokenType::LEFTSHIFTEQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::LEFTSHIFT, expression, source_location);
			} break;
			case Token::TokenType::RIGHTSHIFTEQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::RIGHTSHIFT, expression, source_location);
			} break;
			case Token::TokenType::DOUBLESTAREQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::EXP, expression, source_location);
			} break;
			case Token::TokenType::DOUBLESLASHEQUAL: {
				return std::make_shared<AugAssign>(
					target, BinaryOpType::FLOORDIV, expression, source_location);
			} break;
			default:
				ASSERT(false && "!ICE!: unhandled token in unary operation parsing");
			}
		}
		return {};
	}
};

template<> struct traits<struct ReturnStatementPattern>
{
	using result_type = std::shared_ptr<Return>;
};

struct ReturnStatementPattern : PatternV2<ReturnStatementPattern>
{
	using ResultType = typename traits<ReturnStatementPattern>::result_type;

	// return_stmt:
	// 		| 'return' [star_expressions]
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ReturnPattern>,
			ZeroOrOnePatternV2<StarExpressionsPattern>>;
		if (auto result = pattern1::match(p)) {
			auto [return_token, expressions] = *result;
			if (expressions.has_value()) {
				return std::make_shared<Return>(*expressions,
					SourceLocation{ return_token.start(), (*expressions)->source_location().end });
			}
			return std::make_shared<Return>(
				nullptr, SourceLocation{ return_token.start(), return_token.end() });
		}
		return {};
	}
};

template<> struct traits<struct DottedNamePattern>
{
	using result_type = std::vector<Token>;
};


struct DottedNamePattern : PatternV2<DottedNamePattern>
{
	using ResultType = typename traits<DottedNamePattern>::result_type;

	// dotted_name:
	//     | dotted_name '.' NAME
	//     | NAME
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("dotted_name");
		using pattern1 = PatternMatchV2<DottedNamePattern,
			SingleTokenPatternV2<Token::TokenType::DOT>,
			SingleTokenPatternV2<Token::TokenType::NAME>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("dotted_name '.' NAME");
			auto [dotted_name, _, name] = *result;
			dotted_name.push_back(name.token);
			return dotted_name;
		}

		using pattern2 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::NAME>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("dotted_name '.' NAME");
			auto [name] = *result;
			return { { name.token } };
		}
		return {};
	}
};

template<> struct traits<struct DottedAsNamePattern>
{
	using result_type = std::pair<std::vector<Token>, std::optional<Token>>;
};

struct DottedAsNamePattern : PatternV2<DottedAsNamePattern>
{
	using ResultType = typename traits<DottedAsNamePattern>::result_type;

	// dotted_as_name:
	//     | dotted_name ['as' NAME ]
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("dotted_as_name");
		using pattern1 = PatternMatchV2<DottedNamePattern,
			ZeroOrOnePatternV2<
				AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, AsKeywordPattern>,
				NAMEPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("dotted_name ['as' NAME ]");
			auto [dotted_name, as_name_group] = *result;
			if (as_name_group.has_value()) {
				auto [_, as_name] = *as_name_group;
				return { { dotted_name, as_name.token } };
			}
			return { { dotted_name, {} } };
		}
		return {};
	}
};

template<> struct traits<struct DottedAsNamesPattern>
{
	using result_type = std::vector<typename traits<DottedAsNamePattern>::result_type>;
};


struct DottedAsNamesPattern : PatternV2<DottedAsNamesPattern>
{
	using ResultType = typename traits<DottedAsNamesPattern>::result_type;

	// dotted_as_names:
	//     | ','.dotted_as_name+
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("dotted_as_names");
		using pattern1 = PatternMatchV2<ApplyInBetweenPatternV2<DottedAsNamePattern,
			SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("','.dotted_as_name+");
			auto [dotted_as_name] = *result;
			return dotted_as_name;
		}
		return {};
	}
};

template<> struct traits<struct ImportNamePattern>
{
	using result_type = std::shared_ptr<Import>;
};

struct ImportNamePattern : PatternV2<ImportNamePattern>
{
	using ResultType = typename traits<ImportNamePattern>::result_type;

	// import_name: 'import' dotted_as_names
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("import_name");

		// 'import' dotted_as_names
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ImportKeywordPattern>,
			DottedAsNamesPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'import' dotted_as_names");
			auto [import_token, dotted_as_names] = *result;
			std::vector<alias> aliases;
			aliases.reserve(dotted_as_names.size());
			std::transform(dotted_as_names.begin(),
				dotted_as_names.end(),
				std::back_inserter(aliases),
				[](const auto &el) -> alias {
					auto [name, as_name] = el;
					auto names = std::accumulate(name.begin() + 1,
						name.end(),
						std::string{ name.begin()->start().pointer_to_program,
							name.begin()->end().pointer_to_program },
						[](const std::string &acc, Token t) {
							std::string n{ t.start().pointer_to_program,
								t.end().pointer_to_program };
							return acc + "." + std::move(n);
						});
					if (as_name.has_value()) {
						std::string as_name_{ as_name->start().pointer_to_program,
							as_name->end().pointer_to_program };
						return { names, as_name_ };
					} else {
						return { names, "" };
					}
				});
			return std::make_shared<Import>(std::move(aliases),
				SourceLocation{ import_token.start(),
					dotted_as_names.back().second.has_value()
						? dotted_as_names.back().second->end()
						: dotted_as_names.back().first.back().end() });
		}
		return {};
	}
};
template<> struct traits<struct ImportFromAsNamePattern>
{
	using result_type = std::pair<Token, std::optional<Token>>;
};

struct ImportFromAsNamePattern : PatternV2<ImportFromAsNamePattern>
{
	using ResultType = typename traits<ImportFromAsNamePattern>::result_type;

	// import_from_as_name:
	//     | NAME ['as' NAME ]
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("import_from_as_name");
		// NAME ['as' NAME ]
		using pattern1 = PatternMatchV2<NAMEPattern,
			ZeroOrOnePatternV2<
				AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, AsKeywordPattern>,
				SingleTokenPatternV2<Token::TokenType::NAME>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("NAME ['as' NAME ]");
			auto [name_token, as_name_token_] = *result;
			if (as_name_token_.has_value()) {
				auto [_, as_name_token] = *as_name_token_;
				return { { name_token.token, as_name_token.token } };
			}
			return { { name_token.token, {} } };
		}

		return {};
	}
};

template<> struct traits<struct ImportFromAsNamesPattern>
{
	using result_type = std::vector<typename traits<ImportFromAsNamePattern>::result_type>;
};

struct ImportFromAsNamesPattern : PatternV2<ImportFromAsNamesPattern>
{
	using ResultType = typename traits<ImportFromAsNamesPattern>::result_type;

	// import_from_as_names:
	//     | ','.import_from_as_name+
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("import_from_as_names");

		// ','.import_from_as_name+
		using pattern1 = PatternMatchV2<ApplyInBetweenPatternV2<ImportFromAsNamePattern,
			SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("','.import_from_as_name+");
			auto [import_from_as_name] = *result;
			return import_from_as_name;
		}

		return {};
	}
};

template<> struct traits<struct ImportFromTargetsPattern>
{
	using result_type = typename traits<ImportFromAsNamesPattern>::result_type;
};

struct ImportFromTargetsPattern : PatternV2<ImportFromTargetsPattern>
{
	using ResultType = typename traits<ImportFromTargetsPattern>::result_type;

	// import_from_targets:
	//     | '(' import_from_as_names [','] ')'
	//     | import_from_as_names !','
	//     | '*'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("import_from_targets");

		// '(' import_from_as_names [','] ')'
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LPAREN>,
			ImportFromAsNamesPattern,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'(' import_from_as_names [','] ')'");
			auto [l, import_from_as_names, _, r] = *result;
			(void)l;
			(void)r;
			return import_from_as_names;
		}

		// import_from_as_names !','
		using pattern2 = PatternMatchV2<ImportFromAsNamesPattern,
			NegativeLookAheadV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("import_from_as_names !','");
			auto [import_from_as_names, _] = *result;
			return import_from_as_names;
		}

		// '*'
		using pattern3 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::STAR>>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("'*'");
			auto [star_token] = *result;
			return { { { star_token.token, std::nullopt } } };
		}

		return {};
	}
};

template<> struct traits<struct ImportFromPattern>
{
	using result_type = std::shared_ptr<ImportFrom>;
};

struct ImportFromPattern : PatternV2<ImportFromPattern>
{
	using ResultType = typename traits<ImportFromPattern>::result_type;

	// import_from:
	// | 'from' ('.' | '...')* dotted_name 'import' import_from_targets
	// | 'from' ('.' | '...')+ 'import' import_from_targets
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("import_from");

		// 'from' ('.' | '...')* dotted_name 'import' import_from_targets
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, FromKeywordPattern>,
			ZeroOrMorePatternV2<OrPatternV2<SingleTokenPatternV2<Token::TokenType::DOT>,
				SingleTokenPatternV2<Token::TokenType::ELLIPSIS>>>,
			DottedNamePattern,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ImportKeywordPattern>,
			ImportFromTargetsPattern>;
		if (auto result = pattern1::match(p)) {
			auto [from_token, dots, dotted_name, _, import_from_targets] = *result;
			std::string module = std::accumulate(dotted_name.begin() + 1,
				dotted_name.end(),
				std::string{ dotted_name.begin()->start().pointer_to_program,
					dotted_name.begin()->end().pointer_to_program },
				[](const std::string &acc, Token el) {
					std::string el_{ el.start().pointer_to_program, el.end().pointer_to_program };
					return acc + "." + std::move(el_);
				});
			std::vector<alias> aliases;
			aliases.reserve(import_from_targets.size());
			std::transform(import_from_targets.begin(),
				import_from_targets.end(),
				std::back_inserter(aliases),
				[](const auto &el) -> alias {
					auto [name_token, as_name_token] = el;
					std::string name{ name_token.start().pointer_to_program,
						name_token.end().pointer_to_program };
					if (as_name_token.has_value()) {
						std::string as_name{ as_name_token->start().pointer_to_program,
							as_name_token->end().pointer_to_program };
						return { name, as_name };
					} else {
						return { name, "" };
					}
				});
			const size_t level =
				std::accumulate(dots.begin(), dots.end(), size_t{ 0 }, [](size_t acc, auto t) {
					if (std::holds_alternative<TokenResult<Token::TokenType::DOT>>(t)) {
						return acc + 1;
					} else {
						return acc + 3;
					}
				});

			return std::make_shared<ImportFrom>(module,
				std::move(aliases),
				level,
				SourceLocation{ from_token.start(),
					import_from_targets.back().second.has_value()
						? import_from_targets.back().second->end()
						: import_from_targets.back().first.end() });
		}

		// 'from' ('.' | '...')+ 'import' import_from_targets
		using pattern2 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, FromKeywordPattern>,
			ZeroOrMorePatternV2<OrPatternV2<SingleTokenPatternV2<Token::TokenType::DOT>,
				SingleTokenPatternV2<Token::TokenType::ELLIPSIS>>>,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ImportKeywordPattern>,
			ImportFromTargetsPattern>;
		if (auto result = pattern2::match(p)) {
			auto [from_token, dots, _, import_from_targets] = *result;
			std::string module{};
			std::vector<alias> aliases;
			aliases.reserve(import_from_targets.size());
			std::transform(import_from_targets.begin(),
				import_from_targets.end(),
				std::back_inserter(aliases),
				[](const auto &el) -> alias {
					auto [name_token, as_name_token] = el;
					std::string name{ name_token.start().pointer_to_program,
						name_token.end().pointer_to_program };
					if (as_name_token.has_value()) {
						std::string as_name{ as_name_token->start().pointer_to_program,
							as_name_token->end().pointer_to_program };
						return { name, as_name };
					} else {
						return { name, "" };
					}
				});
			const size_t level =
				std::accumulate(dots.begin(), dots.end(), size_t{ 0 }, [](size_t acc, auto t) {
					if (std::holds_alternative<TokenResult<Token::TokenType::DOT>>(t)) {
						return acc + 1;
					} else {
						return acc + 3;
					}
				});

			return std::make_shared<ImportFrom>(module,
				std::move(aliases),
				level,
				SourceLocation{ from_token.start(),
					import_from_targets.back().second.has_value()
						? import_from_targets.back().second->end()
						: import_from_targets.back().first.end() });
		}

		return {};
	}
};

template<> struct traits<struct ImportStatementPattern>
{
	using result_type = std::shared_ptr<ImportBase>;
};

struct ImportStatementPattern : PatternV2<ImportStatementPattern>
{
	using ResultType = typename traits<ImportStatementPattern>::result_type;

	// import_stmt:
	// 		| import_name
	//		| import_from
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("import_stmt");
		using pattern1 = PatternMatchV2<ImportNamePattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("import_name");
			auto [import_name] = *result;
			return import_name;
		}
		using pattern2 = PatternMatchV2<ImportFromPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("import_from");
			auto [import_from] = *result;
			return import_from;
		}
		return {};
	}
};

template<> struct traits<struct RaiseStatementPattern>
{
	using result_type = std::shared_ptr<Raise>;
};

struct RaiseStatementPattern : PatternV2<RaiseStatementPattern>
{
	using ResultType = typename traits<RaiseStatementPattern>::result_type;

	// raise_stmt:
	//     | 'raise' expression ['from' expression ]
	//     | 'raise'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("raise_stmt");
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, RaiseKeywordPattern>,
			ExpressionPattern,
			ZeroOrOnePatternV2<
				AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, FromKeywordPattern>,
				ExpressionPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'raise' expression ['from' expression ] ");
			auto [raise_token, exception, cause_] = *result;
			if (!cause_.has_value()) {
				return std::make_shared<Raise>(exception,
					nullptr,
					SourceLocation{ raise_token.start(), exception->source_location().end });
			} else {
				auto [_, cause] = *cause_;
				return std::make_shared<Raise>(exception,
					cause,
					SourceLocation{ raise_token.start(), cause->source_location().end });
			}
		}

		using pattern2 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, RaiseKeywordPattern>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'raise'");
			auto [raise_token] = *result;
			return std::make_shared<Raise>(
				SourceLocation{ raise_token.start(), raise_token.end() });
		}
		return {};
	}
};

template<> struct traits<struct DeleteTargetPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

template<> struct traits<struct DeleteTargetsPattern>
{
	using result_type = std::pair<std::vector<std::shared_ptr<ASTNode>>, std::optional<Token>>;
};

template<> struct traits<struct DeleteAtomPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct DeleteAtomPattern : PatternV2<DeleteAtomPattern>
{
	using ResultType = typename traits<DeleteAtomPattern>::result_type;

	// del_t_atom:
	//     | NAME
	//     | '(' del_target ')'
	//     | '(' [del_targets] ')'
	//     | '[' [del_targets] ']'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("del_t_atom");

		// NAME
		using pattern1 = PatternMatchV2<NAMEPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("NAME");
			auto [name_token] = *result;
			std::string name_str{ name_token.token.start().pointer_to_program,
				name_token.token.end().pointer_to_program };
			return std::make_shared<Name>(name_str,
				ContextType::DELETE,
				SourceLocation{ name_token.token.start(), name_token.token.end() });
		}

		// '(' del_target ')'
		using pattern2 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LPAREN>,
			DeleteTargetPattern,
			SingleTokenPatternV2<Token::TokenType::RPAREN>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'(' del_target ')'");
			auto [l, del_target, r] = *result;
			return std::make_shared<Tuple>(std::vector{ del_target },
				ContextType::DELETE,
				SourceLocation{ l.token.start(), r.token.end() });
		}

		// '(' [del_targets] ')'
		using pattern3 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LPAREN>,
			ZeroOrOnePatternV2<DeleteTargetsPattern>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("'(' [del_targets] ')'");
			auto [l, del_targets, r] = *result;
			if (del_targets) {
				return std::make_shared<Tuple>(del_targets->first,
					ContextType::DELETE,
					SourceLocation{ l.token.start(), r.token.end() });
			}
			return std::make_shared<Tuple>(
				ContextType::DELETE, SourceLocation{ l.token.start(), r.token.end() });
		}

		// '[' [del_targets] ']'
		using pattern4 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::LSQB>,
			ZeroOrOnePatternV2<DeleteTargetsPattern>,
			SingleTokenPatternV2<Token::TokenType::RSQB>>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("'[' [del_targets] ']'");
			auto [l, del_targets, r] = *result;
			if (del_targets) {
				return std::make_shared<List>(del_targets->first,
					ContextType::DELETE,
					SourceLocation{ l.token.start(), r.token.end() });
			}
			return std::make_shared<List>(
				ContextType::DELETE, SourceLocation{ l.token.start(), r.token.end() });
		}

		return {};
	}
};

struct DeleteTargetPattern : PatternV2<DeleteTargetPattern>
{
	using ResultType = typename traits<DeleteTargetPattern>::result_type;

	// del_target:
	//     | t_primary '.' NAME !t_lookahead
	//     | t_primary '[' slices ']' !t_lookahead
	//     | del_t_atom
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// t_primary '.' NAME !t_lookahead
		using pattern1 = PatternMatchV2<TPrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::DOT>,
			SingleTokenPatternV2<Token::TokenType::NAME>,
			NegativeLookAheadV2<TLookahead>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("t_primary '.' NAME !t_lookahead");
			auto [value, _, name_token, lookahead] = *result;
			(void)lookahead;
			std::string name{ name_token.token.start().pointer_to_program,
				name_token.token.end().pointer_to_program };
			return std::make_shared<Attribute>(value,
				name,
				ContextType::DELETE,
				SourceLocation{ value->source_location().start, name_token.token.end() });
		}

		// t_primary '[' slices ']' !t_lookahead
		using pattern2 = PatternMatchV2<TPrimaryPattern,
			SingleTokenPatternV2<Token::TokenType::LSQB>,
			SlicesPattern,
			SingleTokenPatternV2<Token::TokenType::RSQB>,
			NegativeLookAheadV2<TLookahead>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("t_primary '[' slices ']' !t_lookahead");
			auto [value, _, slices, r, lookahead] = *result;
			return std::make_shared<Subscript>(value,
				slices,
				ContextType::DELETE,
				SourceLocation{ value->source_location().start, r.token.end() });
		}

		// del_t_atom
		using pattern3 = PatternMatchV2<DeleteAtomPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("del_t_atom");
			auto [del_t_atom] = *result;
			return del_t_atom;
		}
		return {};
	}
};

struct DeleteTargetsPattern : PatternV2<DeleteTargetsPattern>
{
	using ResultType = typename traits<DeleteTargetsPattern>::result_type;

	// del_targets: ','.del_target+ [',']
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("del_targets");

		using pattern1 = PatternMatchV2<ApplyInBetweenPatternV2<DeleteTargetPattern,
											SingleTokenPatternV2<Token::TokenType::COMMA>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("','.del_target+ [',']");
			auto [to_delete, optional_token] = *result;
			return { { to_delete,
				optional_token.has_value() ? std::make_optional(optional_token->token)
										   : std::nullopt } };
		}
		return {};
	}
};

template<> struct traits<struct DeleteStatementPattern>
{
	using result_type = std::shared_ptr<Delete>;
};

struct DeleteStatementPattern : PatternV2<DeleteStatementPattern>
{
	using ResultType = typename traits<DeleteStatementPattern>::result_type;

	// del_stmt:
	// 	| 'del' del_targets &(';' | NEWLINE)
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("'del_stmt'");
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, DeleteKeywordPattern>,
			DeleteTargetsPattern,
			LookAheadV2<OrPatternV2<SingleTokenPatternV2<Token::TokenType::SEMI>,
				SingleTokenPatternV2<Token::TokenType::NEWLINE>>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'del' del_targets &(';' | NEWLINE)");
			auto [delete_token, del_targets, _] = *result;
			return std::make_shared<Delete>(del_targets.first,
				SourceLocation{ delete_token.start(),
					del_targets.second.has_value()
						? del_targets.second->end()
						: del_targets.first.back()->source_location().end });
		}
		return {};
	}
};

template<> struct traits<struct YieldStatementPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct YieldStatementPattern : PatternV2<YieldStatementPattern>
{
	using ResultType = typename traits<YieldStatementPattern>::result_type;

	// yield_stmt: yield_expr
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("yield_stmt");

		using pattern1 = PatternMatchV2<YieldExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("yield_expr");
			auto [yield_expr] = *result;
			return yield_expr;
		}
		return {};
	}
};

template<> struct traits<struct AssertStatementPattern>
{
	using result_type = std::shared_ptr<Assert>;
};

struct AssertStatementPattern : PatternV2<AssertStatementPattern>
{
	using ResultType = typename traits<AssertStatementPattern>::result_type;

	// assert_stmt: 'assert' expression [',' expression ]
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("assert_stmt");

		// 'assert' expression [',' expression ]
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, AssertKeywordPattern>,
			ExpressionPattern,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>, ExpressionPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("assert_stmt: 'assert' expression [',' expression ]");
			auto [assert_token, test, msg_] = *result;

			if (msg_.has_value()) {
				auto [_, msg] = *msg_;
				return std::make_shared<Assert>(
					test, msg, SourceLocation{ assert_token.start(), msg->source_location().end });
			}
			return std::make_shared<Assert>(
				test, nullptr, SourceLocation{ assert_token.start(), test->source_location().end });
		}

		return {};
	}
};

template<> struct traits<struct GlobalStatementPattern>
{
	using result_type = std::shared_ptr<Global>;
};

struct GlobalStatementPattern : PatternV2<GlobalStatementPattern>
{
	using ResultType = typename traits<GlobalStatementPattern>::result_type;

	// global_stmt: 'global' ','.NAME+
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("GlobalStatementPattern");
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, GlobalKeywordPattern>,
			ApplyInBetweenPatternV2<NAMEPattern, SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'global' ','.NAME+");
			auto [global_token, name_tokens] = *result;
			std::vector<std::string> names;
			names.reserve(name_tokens.size());
			std::transform(name_tokens.begin(),
				name_tokens.end(),
				std::back_inserter(names),
				[](const auto &el) -> std::string {
					return { el.token.start().pointer_to_program,
						el.token.end().pointer_to_program };
				});
			return std::make_shared<Global>(
				names, SourceLocation{ global_token.start(), name_tokens.back().token.end() });
		}

		return {};
	}
};

template<> struct traits<struct NonLocalStatementPattern>
{
	using result_type = std::shared_ptr<NonLocal>;
};

struct NonLocalStatementPattern : PatternV2<NonLocalStatementPattern>
{
	using ResultType = typename traits<NonLocalStatementPattern>::result_type;

	// nonlocal_stmt: 'nonlocal' ','.NAME+
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("NonLocalStatementPattern");
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, NonLocalKeywordPattern>,
			ApplyInBetweenPatternV2<NAMEPattern, SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'nonlocal' ','.NAME+");
			auto [nonlocal_token, name_tokens] = *result;
			std::vector<std::string> names;
			names.reserve(name_tokens.size());
			std::transform(name_tokens.begin(),
				name_tokens.end(),
				std::back_inserter(names),
				[](const auto &el) -> std::string {
					return { el.token.start().pointer_to_program,
						el.token.end().pointer_to_program };
				});
			return std::make_shared<NonLocal>(
				names, SourceLocation{ nonlocal_token.start(), name_tokens.back().token.end() });
		}

		return {};
	}
};


template<> struct traits<struct SmallStatementPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct SmallStatementPattern : PatternV2<SmallStatementPattern>
{
	using ResultType = typename traits<SmallStatementPattern>::result_type;

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
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// assignment
		using pattern1 = PatternMatchV2<AssignmentPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("assignment");
			auto [assignment] = *result;
			return assignment;
		}

		// star_expressions
		using pattern2 = PatternMatchV2<StarExpressionsPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("star_expressions");
			auto [expression] = *result;
			return expression;
		}

		// return_stmt
		using pattern3 = PatternMatchV2<ReturnStatementPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("return_stmt");
			auto [stmt] = *result;
			return stmt;
		}

		// import_stmt
		using pattern4 = PatternMatchV2<ImportStatementPattern>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("import_stmt");
			auto [import_stmt] = *result;
			return import_stmt;
		}

		// raise_stmt
		using pattern5 = PatternMatchV2<RaiseStatementPattern>;
		if (auto result = pattern5::match(p)) {
			DEBUG_LOG("raise_stmt");
			auto [raise_stmt] = *result;
			return raise_stmt;
		}

		// 'pass'
		using pattern6 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, PassKeywordPattern>>;
		if (auto result = pattern6::match(p)) {
			DEBUG_LOG("pass");
			auto [pass_token] = *result;
			return std::make_shared<Pass>(SourceLocation{ pass_token.start(), pass_token.end() });
		}

		// del_stmt
		using pattern7 = PatternMatchV2<DeleteStatementPattern>;
		if (auto result = pattern7::match(p)) {
			DEBUG_LOG("del_stmt");
			auto [del_stmt] = *result;
			return del_stmt;
		}

		// yield_stmt
		using pattern8 = PatternMatchV2<YieldStatementPattern>;
		if (auto result = pattern8::match(p)) {
			DEBUG_LOG("yield_stmt");
			auto [yield_stmt] = *result;
			return yield_stmt;
		}

		// assert_stmt
		using pattern9 = PatternMatchV2<AssertStatementPattern>;
		if (auto result = pattern9::match(p)) {
			DEBUG_LOG("assert_stmt");
			auto [assert_stmt] = *result;
			return assert_stmt;
		}

		// 'break'
		using pattern10 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, BreakKeywordPattern>>;
		if (auto result = pattern10::match(p)) {
			DEBUG_LOG("'break'");
			auto [break_token] = *result;
			return std::make_shared<Break>(
				SourceLocation{ break_token.start(), break_token.end() });
		}

		// 'continue'
		using pattern11 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ContinueKeywordPattern>>;
		if (auto result = pattern11::match(p)) {
			DEBUG_LOG("'continue'");
			auto [continue_token] = *result;
			return std::make_shared<Continue>(
				SourceLocation{ continue_token.start(), continue_token.end() });
		}

		// global_stmt
		using pattern12 = PatternMatchV2<GlobalStatementPattern>;
		if (auto result = pattern12::match(p)) {
			DEBUG_LOG("global_stmt");
			auto [global_stmt] = *result;
			return global_stmt;
		}

		// nonlocal_stmt
		using pattern13 = PatternMatchV2<NonLocalStatementPattern>;
		if (auto result = pattern13::match(p)) {
			DEBUG_LOG("nonlocal_stmt");
			auto [nonlocal_stmt] = *result;
			return nonlocal_stmt;
		}
		return {};
	}
};

template<> struct traits<struct SimpleStatementPattern>
{
	using result_type = std::vector<std::shared_ptr<ASTNode>>;
};

struct SimpleStatementPattern : PatternV2<SimpleStatementPattern>
{
	using ResultType = typename traits<SimpleStatementPattern>::result_type;

	// simple_stmt:
	// 	| small_stmt !';' NEWLINE  # Not needed, there for speedup
	// 	| ';'.small_stmt+ [';'] NEWLINE
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// small_stmt !';' NEWLINE
		using pattern1 = PatternMatchV2<SmallStatementPattern,
			NegativeLookAheadV2<SingleTokenPatternV2<Token::TokenType::SEMI>>,
			SingleTokenPatternV2<Token::TokenType::NEWLINE>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("small_stmt !';' NEWLINE");
			auto [small_stmt, s, _] = *result;
			(void)s;
			return { { small_stmt } };
		}

		// ';'.small_stmt+ [';'] NEWLINE
		using pattern2 = PatternMatchV2<ApplyInBetweenPatternV2<SmallStatementPattern,
											SingleTokenPatternV2<Token::TokenType::SEMI>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::SEMI>>,
			SingleTokenPatternV2<Token::TokenType::NEWLINE>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("';'.small_stmt+ [';'] NEWLINE");
			auto [small_stmts, s, _] = *result;
			(void)s;
			return small_stmts;
		}

		return {};
	}
};

template<> struct traits<struct AnnotationPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct AnnotationPattern : PatternV2<AnnotationPattern>
{
	using ResultType = typename traits<AnnotationPattern>::result_type;

	// annotation: ':' expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("annotation");
		// ':' expression
		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::COLON>, ExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("':' expression");
			auto [_, expression] = *result;
			return expression;
		}

		return {};
	}
};

template<> struct traits<struct ParamPattern>
{
	using result_type = std::shared_ptr<Argument>;
};

struct ParamPattern : PatternV2<ParamPattern>
{
	using ResultType = typename traits<ParamPattern>::result_type;

	// param: NAME annotation?
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("param");

		using pattern1 = PatternMatchV2<NAMEPattern, ZeroOrOnePatternV2<AnnotationPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("NAME annotation?");
			auto [name_token, annotation] = *result;
			std::string argname{ name_token.token.start().pointer_to_program,
				name_token.token.end().pointer_to_program };

			return std::make_shared<Argument>(argname,
				annotation.value_or(nullptr),
				"",
				SourceLocation{ name_token.token.start(),
					annotation.has_value() ? (*annotation)->source_location().end
										   : name_token.token.end() });
		}

		return {};
	}
};

template<> struct traits<struct ParamNoDefaultPattern>
{
	using result_type = std::shared_ptr<Argument>;
};

struct ParamNoDefaultPattern : PatternV2<ParamNoDefaultPattern>
{
	using ResultType = typename traits<ParamNoDefaultPattern>::result_type;

	// param_no_default:
	//     | param ',' TYPE_COMMENT?
	//     | param TYPE_COMMENT? &')'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// param ',' TYPE_COMMENT?
		// TODO: implement TYPE_COMMENT
		using pattern1 =
			PatternMatchV2<ParamPattern, SingleTokenPatternV2<Token::TokenType::COMMA>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("param ',' TYPE_COMMENT?");
			auto [param, _] = *result;
			return param;
		}

		// param TYPE_COMMENT? &')'
		// TODO: implement TYPE_COMMENT
		using pattern2 = PatternMatchV2<ParamPattern,
			LookAheadV2<SingleTokenPatternV2<Token::TokenType::RPAREN>>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("param TYPE_COMMENT? &')'");
			auto [param, _] = *result;
			return param;
		}

		return {};
	}
};

struct DefaultPattern : PatternV2<DefaultPattern>
{
	using ResultType = typename traits<DefaultPattern>::result_type;

	// default: '=' expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("default");
		using pattern1 =
			PatternMatchV2<SingleTokenPatternV2<Token::TokenType::EQUAL>, ExpressionPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'=' expression");
			auto [_, expression] = *result;
			return expression;
		}

		return {};
	}
};

template<> struct traits<struct ParamWithDefaultPattern>
{
	using result_type = std::pair<std::shared_ptr<Argument>, std::shared_ptr<ASTNode>>;
};

struct ParamWithDefaultPattern : PatternV2<ParamWithDefaultPattern>
{
	using ResultType = typename traits<ParamWithDefaultPattern>::result_type;

	// param_with_default:
	//     | param default ',' TYPE_COMMENT?
	//     | param default TYPE_COMMENT? &')'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("param_with_default");

		using pattern1 = PatternMatchV2<ParamPattern,
			DefaultPattern,
			SingleTokenPatternV2<Token::TokenType::COMMA>>;
		// param default ',' TYPE_COMMENT?
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("param default ',' TYPE_COMMENT?");
			auto [param, default_, _] = *result;
			return { { param, default_ } };
		}

		using pattern2 = PatternMatchV2<ParamPattern,
			DefaultPattern,
			LookAheadV2<SingleTokenPatternV2<Token::TokenType::RPAREN>>>;
		// param default TYPE_COMMENT? &')'
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("param default TYPE_COMMENT? &')'");
			auto [param, default_, _] = *result;
			return { { param, default_ } };
		}

		return {};
	}
};

template<> struct traits<struct ParamMaybeDefaultPattern>
{
	using result_type =
		std::pair<std::shared_ptr<Argument>, std::optional<std::shared_ptr<ASTNode>>>;
};

struct ParamMaybeDefaultPattern : PatternV2<ParamMaybeDefaultPattern>
{
	using ResultType = typename traits<ParamMaybeDefaultPattern>::result_type;

	// param_maybe_default:
	//     | param default? ',' TYPE_COMMENT?
	//     | param default? TYPE_COMMENT? &')'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("param_maybe_default");

		using pattern1 = PatternMatchV2<ParamPattern,
			ZeroOrOnePatternV2<DefaultPattern>,
			SingleTokenPatternV2<Token::TokenType::COMMA>>;
		// param default? ',' TYPE_COMMENT?
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("param default? ',' TYPE_COMMENT?");
			auto [param, default_, _] = *result;
			if (default_.has_value()) { return { { param, *default_ } }; }
			return { { param, {} } };
		}

		using pattern2 = PatternMatchV2<ParamPattern,
			ZeroOrOnePatternV2<DefaultPattern>,
			LookAheadV2<SingleTokenPatternV2<Token::TokenType::RPAREN>>>;
		// param default? TYPE_COMMENT? &')'
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("param default? TYPE_COMMENT? &')'");
			auto [param, default_, _] = *result;
			if (default_.has_value()) { return { { param, *default_ } }; }
			return { { param, {} } };
		}

		return {};
	}
};

template<> struct traits<struct KeywordsPattern>
{
	using result_type = std::shared_ptr<Argument>;
};

struct KeywordsPattern : PatternV2<KeywordsPattern>
{
	using ResultType = typename traits<KeywordsPattern>::result_type;

	// kwds: '**' param_no_default
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("KeywordsPattern");
		// '**' param_no_default
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::DOUBLESTAR>,
			ParamNoDefaultPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'**'param_no_default");
			auto [token, param_no_default] = *result;
			return std::make_shared<Argument>(param_no_default->name(),
				param_no_default->annotation(),
				"",
				SourceLocation{ token.token.start(), param_no_default->source_location().end });
		}

		return {};
	}
};

template<> struct traits<struct StarEtcPattern>
{
	using result_type = std::tuple<std::optional<std::shared_ptr<Argument>>,
		std::vector<std::pair<std::shared_ptr<Argument>, std::optional<std::shared_ptr<ASTNode>>>>,
		std::optional<std::shared_ptr<Argument>>>;
};

struct StarEtcPattern : PatternV2<StarEtcPattern>
{
	using ResultType = typename traits<StarEtcPattern>::result_type;

	// star_etc:
	//     | '*' param_no_default param_maybe_default* [kwds]
	//     | '*' ',' param_maybe_default+ [kwds]
	//     | kwds
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("star_etc");
		// '*' param_no_default param_maybe_default* [kwds]
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::STAR>,
			ParamNoDefaultPattern,
			ZeroOrMorePatternV2<ParamMaybeDefaultPattern>,
			ZeroOrOnePatternV2<KeywordsPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'*' param_no_default param_maybe_default*");

			auto [_, param_no_default, param_maybe_defaults, kwds] = *result;

			return { { param_no_default, param_maybe_defaults, kwds } };
		}

		// '*' ',' param_maybe_default+ [kwds]
		using pattern2 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::STAR>,
			SingleTokenPatternV2<Token::TokenType::COMMA>,
			ZeroOrMorePatternV2<ParamMaybeDefaultPattern>,
			ZeroOrOnePatternV2<KeywordsPattern>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'*' ',' param_maybe_default+");
			auto [s, _, param_maybe_defaults, kwds] = *result;
			(void)s;
			return { { {}, param_maybe_defaults, kwds } };
		}

		// kwds
		using pattern3 = PatternMatchV2<KeywordsPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("kwds");
			auto [kwds] = *result;
			return { { {}, {}, kwds } };
		}

		return {};
	}
};

template<> struct traits<struct SlashWithDefaultPattern>
{
	using result_type = std::pair<std::vector<typename traits<ParamNoDefaultPattern>::result_type>,
		std::vector<typename traits<ParamWithDefaultPattern>::result_type>>;
};

struct SlashWithDefaultPattern : PatternV2<SlashWithDefaultPattern>
{
	using ResultType = typename traits<SlashWithDefaultPattern>::result_type;

	// slash_with_default:
	//     | param_no_default* param_with_default+ '/' ','
	//     | param_no_default* param_with_default+ '/' &')'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<ZeroOrMorePatternV2<ParamNoDefaultPattern>,
			OneOrMorePatternV2<ParamWithDefaultPattern>,
			SingleTokenPatternV2<Token::TokenType::SLASH>,
			SingleTokenPatternV2<Token::TokenType::COMMA>>;
		// param_no_default* param_with_default+ '/' ','
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("param_no_default* param_with_default+ '/' ','");
			auto [params_no_default, param_with_default, s, _] = *result;
			(void)s;
			return { { params_no_default, param_with_default } };
		}

		using pattern2 = PatternMatchV2<ZeroOrMorePatternV2<ParamNoDefaultPattern>,
			OneOrMorePatternV2<ParamWithDefaultPattern>,
			SingleTokenPatternV2<Token::TokenType::SLASH>,
			LookAheadV2<SingleTokenPatternV2<Token::TokenType::RPAREN>>>;
		// param_no_default* param_with_default+ '/' &')'
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("param_no_default* param_with_default+ '/' &')'");
			auto [params_no_default, param_with_default, s, _] = *result;
			(void)s;
			return { { params_no_default, param_with_default } };
		}

		return {};
	}
};

template<> struct traits<struct SlashNoDefaultPattern>
{
	using result_type = std::vector<std::shared_ptr<Argument>>;
};

struct SlashNoDefaultPattern : PatternV2<SlashNoDefaultPattern>
{
	using ResultType = typename traits<SlashNoDefaultPattern>::result_type;

	// slash_no_default:
	//     | param_no_default+ '/' ','
	//     | param_no_default+ '/' &')'
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<OneOrMorePatternV2<ParamNoDefaultPattern>,
			SingleTokenPatternV2<Token::TokenType::SLASH>,
			SingleTokenPatternV2<Token::TokenType::COMMA>>;
		// param_no_default+ '/' ','
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("param_no_default+ '/' ','");
			auto [params_no_default, s, _] = *result;
			(void)s;
			return params_no_default;
		}

		using pattern2 = PatternMatchV2<OneOrMorePatternV2<ParamNoDefaultPattern>,
			SingleTokenPatternV2<Token::TokenType::SLASH>,
			LookAheadV2<SingleTokenPatternV2<Token::TokenType::RPAREN>>>;
		// param_no_default+ '/' &')'
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("param_no_default+ '/' &')'");
			auto [params_no_default, s, _] = *result;
			(void)s;
			return params_no_default;
		}

		return {};
	}
};

template<> struct traits<struct ParametersPattern>
{
	using result_type = std::shared_ptr<Arguments>;
};

struct ParametersPattern : PatternV2<ParametersPattern>
{
	using ResultType = typename traits<ParametersPattern>::result_type;

	// parameters:
	//     | slash_no_default param_no_default* param_with_default* [star_etc]
	//     | slash_with_default param_with_default* [star_etc]
	//     | param_no_default+ param_with_default* [star_etc]
	//     | param_with_default+ [star_etc]
	//     | star_etc
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("parameters");

		// slash_no_default param_no_default* param_with_default* [star_etc]
		using pattern1 = PatternMatchV2<SlashNoDefaultPattern,
			ZeroOrMorePatternV2<ParamNoDefaultPattern>,
			ZeroOrMorePatternV2<ParamWithDefaultPattern>,
			ZeroOrOnePatternV2<StarEtcPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("slash_no_default param_no_default* param_with_default* [star_etc]");
			auto [slash_no_default, params_no_default, params_with_default, star_etc] = *result;

			std::vector<std::shared_ptr<Argument>> posonlyargs;
			std::vector<std::shared_ptr<Argument>> args;
			std::shared_ptr<Argument> vararg;
			std::vector<std::shared_ptr<Argument>> kwonlyargs;
			std::vector<std::shared_ptr<ASTNode>> kw_defaults;
			std::shared_ptr<Argument> kwarg;
			std::vector<std::shared_ptr<ASTNode>> defaults;
			SourceLocation source_location{
				slash_no_default.front()->source_location().start,
				p.lexer().peek_token(p.token_position() - 1)->end(),// too lazy to figure this out
			};

			posonlyargs.insert(posonlyargs.end(), slash_no_default.begin(), slash_no_default.end());

			args.insert(args.end(), params_no_default.begin(), params_no_default.end());
			for (const auto &[param, default_] : params_with_default) {
				args.push_back(param);
				defaults.push_back(default_);
			}

			if (star_etc.has_value()) {
				auto [vararg_, kwonlyargs_, kwarg_] = *star_etc;
				vararg = vararg_.value_or(nullptr);
				for (const auto &[param, default_] : kwonlyargs_) {
					kwonlyargs.push_back(param);
					kw_defaults.push_back(default_.value_or(nullptr));
				}
				kwarg = kwarg_.value_or(nullptr);
			}

			return std::make_shared<Arguments>(posonlyargs,
				args,
				vararg,
				kwonlyargs,
				kw_defaults,
				kwarg,
				defaults,
				source_location);
		}

		// slash_with_default param_with_default* [star_etc]
		using pattern2 = PatternMatchV2<SlashWithDefaultPattern,
			ZeroOrMorePatternV2<ParamWithDefaultPattern>,
			ZeroOrOnePatternV2<StarEtcPattern>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("slash_with_default param_with_default* [star_etc]");
			auto [posonly_params, params_with_default, star_etc] = *result;
			const auto &[posonly_params_without_default, posonly_params_with_default] =
				posonly_params;

			std::vector<std::shared_ptr<Argument>> posonlyargs;
			std::vector<std::shared_ptr<Argument>> args;
			std::shared_ptr<Argument> vararg;
			std::vector<std::shared_ptr<Argument>> kwonlyargs;
			std::vector<std::shared_ptr<ASTNode>> kw_defaults;
			std::shared_ptr<Argument> kwarg;
			std::vector<std::shared_ptr<ASTNode>> defaults;
			SourceLocation source_location{
				posonly_params_without_default.empty()
					? posonly_params_with_default.front().first->source_location().start
					: posonly_params_without_default.front()->source_location().start,
				p.lexer().peek_token(p.token_position() - 1)->end(),// too lazy to figure this out
			};

			for (const auto &param : posonly_params_without_default) {
				posonlyargs.push_back(param);
			}

			for (const auto &[param, default_value] : posonly_params_with_default) {
				posonlyargs.push_back(param);
				defaults.push_back(default_value);
			}

			for (const auto &[param, default_value] : params_with_default) {
				args.push_back(param);
				defaults.push_back(default_value);
			}

			if (star_etc.has_value()) {
				auto [vararg_, kwonlyargs_, kwarg_] = *star_etc;
				vararg = vararg_.value_or(nullptr);
				for (const auto &[param, default_] : kwonlyargs_) {
					kwonlyargs.push_back(param);
					kw_defaults.push_back(default_.value_or(nullptr));
				}
				kwarg = kwarg_.value_or(nullptr);
			}

			return std::make_shared<Arguments>(posonlyargs,
				args,
				vararg,
				kwonlyargs,
				kw_defaults,
				kwarg,
				defaults,
				source_location);
		}

		// param_no_default+ param_with_default* [star_etc]
		using pattern3 = PatternMatchV2<OneOrMorePatternV2<ParamNoDefaultPattern>,
			ZeroOrMorePatternV2<ParamWithDefaultPattern>,
			ZeroOrOnePatternV2<StarEtcPattern>>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("param_no_default+ param_with_default* [star_etc]");
			auto [params_no_default, params_with_default, star_etc] = *result;

			std::vector<std::shared_ptr<Argument>> posonlyargs;
			std::vector<std::shared_ptr<Argument>> args;
			std::shared_ptr<Argument> vararg;
			std::vector<std::shared_ptr<Argument>> kwonlyargs;
			std::vector<std::shared_ptr<ASTNode>> kw_defaults;
			std::shared_ptr<Argument> kwarg;
			std::vector<std::shared_ptr<ASTNode>> defaults;
			SourceLocation source_location{
				params_no_default.front()->source_location().start,
				p.lexer().peek_token(p.token_position() - 1)->end(),// too lazy to figure this out
			};

			args.insert(args.end(), params_no_default.begin(), params_no_default.end());
			for (const auto &[param, default_] : params_with_default) {
				args.push_back(param);
				defaults.push_back(default_);
			}

			if (star_etc.has_value()) {
				auto [vararg_, kwonlyargs_, kwarg_] = *star_etc;
				vararg = vararg_.value_or(nullptr);
				for (const auto &[param, default_] : kwonlyargs_) {
					kwonlyargs.push_back(param);
					kw_defaults.push_back(default_.value_or(nullptr));
				}
				kwarg = kwarg_.value_or(nullptr);
			}

			return std::make_shared<Arguments>(posonlyargs,
				args,
				vararg,
				kwonlyargs,
				kw_defaults,
				kwarg,
				defaults,
				source_location);
		}

		// param_with_default+ [star_etc]
		using pattern4 = PatternMatchV2<OneOrMorePatternV2<ParamWithDefaultPattern>,
			ZeroOrOnePatternV2<StarEtcPattern>>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("param_with_default+ [star_etc]");
			auto [params_with_default, star_etc] = *result;

			std::vector<std::shared_ptr<Argument>> posonlyargs;
			std::vector<std::shared_ptr<Argument>> args;
			std::shared_ptr<Argument> vararg;
			std::vector<std::shared_ptr<Argument>> kwonlyargs;
			std::vector<std::shared_ptr<ASTNode>> kw_defaults;
			std::shared_ptr<Argument> kwarg;
			std::vector<std::shared_ptr<ASTNode>> defaults;
			SourceLocation source_location{
				params_with_default.front().first->source_location().start,
				p.lexer().peek_token(p.token_position() - 1)->end(),
			};

			for (const auto &[param, default_] : params_with_default) {
				args.push_back(param);
				defaults.push_back(default_);
			}

			if (star_etc.has_value()) {
				auto [vararg_, kwonlyargs_, kwarg_] = *star_etc;
				vararg = vararg_.value_or(nullptr);
				for (const auto &[param, default_] : kwonlyargs_) {
					kwonlyargs.push_back(param);
					kw_defaults.push_back(default_.value_or(nullptr));
				}
				kwarg = kwarg_.value_or(nullptr);
			}

			return std::make_shared<Arguments>(posonlyargs,
				args,
				vararg,
				kwonlyargs,
				kw_defaults,
				kwarg,
				defaults,
				source_location);
		}

		auto start =
			p.lexer().peek_token(p.token_position())->start();// too lazy to figure this out

		// star_etc
		using pattern5 = PatternMatchV2<StarEtcPattern>;
		if (auto result = pattern5::match(p)) {
			DEBUG_LOG("star_etc");
			auto [star_etc] = *result;
			std::vector<std::shared_ptr<Argument>> posonlyargs;
			std::vector<std::shared_ptr<Argument>> args;
			std::shared_ptr<Argument> vararg;
			std::vector<std::shared_ptr<Argument>> kwonlyargs;
			std::vector<std::shared_ptr<ASTNode>> kw_defaults;
			std::shared_ptr<Argument> kwarg;
			std::vector<std::shared_ptr<ASTNode>> defaults;
			SourceLocation source_location{
				start,
				p.lexer().peek_token(p.token_position() - 1)->end(),// too lazy to figure this out
			};

			auto [vararg_, kwonlyargs_, kwarg_] = star_etc;
			vararg = vararg_.value_or(nullptr);
			for (const auto &[param, default_] : kwonlyargs_) {
				kwonlyargs.push_back(param);
				kw_defaults.push_back(default_.value_or(nullptr));
			}
			kwarg = kwarg_.value_or(nullptr);

			return std::make_shared<Arguments>(posonlyargs,
				args,
				vararg,
				kwonlyargs,
				kw_defaults,
				kwarg,
				defaults,
				source_location);
		}
		return {};
	}
};

template<> struct traits<struct ParamsPattern>
{
	using result_type = std::shared_ptr<Arguments>;
};

struct ParamsPattern : PatternV2<ParamsPattern>
{
	using ResultType = typename traits<ParamsPattern>::result_type;

	// params:
	// 		| parameters
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("params");

		// parameters
		using pattern1 = PatternMatchV2<ParametersPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("parameters");
			auto [parameters] = *result;
			return parameters;
		}

		return {};
	}
};

template<> struct traits<struct StatementsPattern>
{
	using result_type = std::vector<std::shared_ptr<ASTNode>>;
};

template<> struct traits<struct BlockPattern>
{
	using result_type = std::vector<std::shared_ptr<ASTNode>>;
};

struct BlockPattern : PatternV2<BlockPattern>
{
	using ResultType = typename traits<BlockPattern>::result_type;

	// 	block:
	//	 	| NEWLINE INDENT statements DEDENT
	// 		| simple_stmt
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<SingleTokenPatternV2<Token::TokenType::NEWLINE>,
			SingleTokenPatternV2<Token::TokenType::INDENT>,
			StatementsPattern,
			SingleTokenPatternV2<Token::TokenType::DEDENT>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("NEWLINE INDENT statements DEDENT");
			auto [n, i, statements, _] = *result;
			return statements;
		}

		using pattern2 = PatternMatchV2<SimpleStatementPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("simple_stmt");
			auto [simple_stmt] = *result;
			return simple_stmt;
		}

		return {};
	}
};

template<> struct traits<struct FunctionNamePattern>
{
	using result_type = std::shared_ptr<Constant>;
};

struct FunctionNamePattern : PatternV2<FunctionNamePattern>
{
	using ResultType = typename traits<FunctionNamePattern>::result_type;

	// function_name: NAME
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("function_name");
		using pattern1 = PatternMatchV2<NAMEPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("NAME");
			auto [token_name] = *result;
			std::string function_name{ token_name.token.start().pointer_to_program,
				token_name.token.end().pointer_to_program };
			return std::make_shared<Constant>(
				function_name, SourceLocation{ token_name.token.start(), token_name.token.end() });
		}
		return {};
	}
};

template<> struct traits<struct FunctionDefinitionRawStatement>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct FunctionDefinitionRawStatement : PatternV2<FunctionDefinitionRawStatement>
{
	using ResultType = typename traits<FunctionDefinitionRawStatement>::result_type;

	// function_def_raw:
	//     | 'def' NAME '(' [params] ')' ['->' expression ] ':' [func_type_comment] block
	//     | ASYNC 'def' NAME '(' [params] ')' ['->' expression ] ':' [func_type_comment] block
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// 'def' NAME '(' [params] ')' ['->' expression ] ':' [func_type_comment] block
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, DefKeywordPattern>,
			SingleTokenPatternV2<Token::TokenType::NAME>,
			SingleTokenPatternV2<Token::TokenType::LPAREN>,
			ZeroOrOnePatternV2<ParamsPattern>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::RARROW>, ExpressionPattern>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			// ZeroOrOnePatternV2<FuncTypeCommentPattern>,
			BlockPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG(
				"'def' NAME '(' [params] ')' ['->' expression ] ':' [func_type_comment] block");
			auto [def_token,
				name,
				l,
				params,
				r,
				return_expression,
				colon_token,
				// func_type_comment,
				body] = *result;
			std::string function_name{ name.token.start().pointer_to_program,
				name.token.end().pointer_to_program };
			return std::make_shared<FunctionDefinition>(function_name,
				params.value_or(
					std::make_shared<Arguments>(SourceLocation{ l.token.start(), r.token.end() })),
				body,
				std::vector<std::shared_ptr<ASTNode>>{},
				return_expression.has_value() ? std::get<1>(*return_expression) : nullptr,
				"",
				SourceLocation{ def_token.start(), body.back()->source_location().end });
		}

		// ASYNC 'def' NAME '(' [params] ')' ['->' expression ] ':' [func_type_comment] block
		using pattern2 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, AsyncKeywordPattern>,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, DefKeywordPattern>,
			SingleTokenPatternV2<Token::TokenType::NAME>,
			SingleTokenPatternV2<Token::TokenType::LPAREN>,
			ZeroOrOnePatternV2<ParamsPattern>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::RARROW>, ExpressionPattern>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			// ZeroOrOnePatternV2<FuncTypeCommentPattern>,
			BlockPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG(
				"ASYNC 'def' NAME '(' [params] ')' ['->' expression ] ':' [func_type_comment] "
				"block");
			auto [async_token,
				def_token,
				name,
				l,
				params,
				r,
				return_expression,
				colon_token,
				// func_type_comment,
				body] = *result;
			std::string function_name{ name.token.start().pointer_to_program,
				name.token.end().pointer_to_program };
			return std::make_shared<AsyncFunctionDefinition>(function_name,
				params.value_or(
					std::make_shared<Arguments>(SourceLocation{ l.token.start(), r.token.end() })),
				body,
				std::vector<std::shared_ptr<ASTNode>>{},
				return_expression.has_value() ? std::get<1>(*return_expression) : nullptr,
				"",
				SourceLocation{ async_token.start(), body.back()->source_location().end });
		}
		return {};
	}
};

template<> struct traits<struct DecoratorsPattern>
{
	using result_type = std::pair<Token, std::vector<std::shared_ptr<ASTNode>>>;
};

struct DecoratorsPattern : PatternV2<DecoratorsPattern>
{
	using ResultType = typename traits<DecoratorsPattern>::result_type;

	// decorators: ('@' named_expression NEWLINE )+
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("DecoratorsPattern");

		// ('@' named_expression NEWLINE )+
		using pattern1 =
			PatternMatchV2<OneOrMorePatternV2<SingleTokenPatternV2<Token::TokenType::AT>,
				NamedExpressionPattern,
				SingleTokenPatternV2<Token::TokenType::NEWLINE>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("('@' named_expression NEWLINE )+");
			auto [decorators] = *result;
			std::vector<std::shared_ptr<ASTNode>> decorator_vector;
			std::optional<Token> first_token;
			for (bool first = true; const auto &decorator : decorators) {
				auto [at_token, named_expression, _] = decorator;
				if (first) { first_token = at_token.token; }
				decorator_vector.push_back(named_expression);
				first = false;
			}
			ASSERT(first_token.has_value());
			return { { *first_token, decorator_vector } };
		}

		return {};
	}
};


template<> struct traits<struct FunctionDefinitionStatementPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct FunctionDefinitionStatementPattern : PatternV2<FunctionDefinitionStatementPattern>
{
	using ResultType = typename traits<FunctionDefinitionStatementPattern>::result_type;

	// function_def:
	//     | decorators function_def_raw
	//     | function_def_raw
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// decorators function_def_raw
		using pattern1 = PatternMatchV2<DecoratorsPattern, FunctionDefinitionRawStatement>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("decorators function_def_raw");
			auto [decorators, function_def] = *result;
			if (auto f = as<FunctionDefinition>(function_def)) {
				return std::make_shared<FunctionDefinition>(f->name(),
					f->args(),
					f->body(),
					decorators.second,
					f->returns(),
					f->type_comment(),
					SourceLocation{ decorators.first.start(), f->source_location().end });
			}
			auto f = as<AsyncFunctionDefinition>(function_def);
			ASSERT(f);
			return std::make_shared<AsyncFunctionDefinition>(f->name(),
				f->args(),
				f->body(),
				decorators.second,
				f->returns(),
				f->type_comment(),
				SourceLocation{ decorators.first.start(), f->source_location().end });
		}

		// function_def_raw
		using pattern2 = PatternMatchV2<FunctionDefinitionRawStatement>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("function_def_raw");
			auto [function_def] = *result;
			return function_def;
		}

		return {};
	}
};

template<> struct traits<struct ElseBlockPattern>
{
	using result_type = std::vector<std::shared_ptr<ASTNode>>;
};

struct ElseBlockPattern : PatternV2<ElseBlockPattern>
{
	using ResultType = typename traits<ElseBlockPattern>::result_type;

	// else_block: 'else' ':' block
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("else_block");

		// else_block: 'else' ':' block
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ElseKeywordPattern>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'else' ':' block");
			auto [e, _, block] = *result;
			return block;
		}

		return {};
	}
};

template<> struct traits<struct ElifStatementPattern>
{
	using result_type = std::shared_ptr<If>;
};

struct ElifStatementPattern : PatternV2<ElifStatementPattern>
{
	using ResultType = typename traits<ElifStatementPattern>::result_type;

	// elif_stmt:
	//     | 'elif' named_expression ':' block elif_stmt
	//     | 'elif' named_expression ':' block [else_block]
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("elif_stmt");

		// 'elif' named_expression ':' block elif_stmt
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ElifKeywordPattern>,
			NamedExpressionPattern,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern,
			ElifStatementPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'elif' named_expression ':' block elif_stmt");
			auto [elif_token, test, _, body, orelse] = *result;
			return std::make_shared<If>(test,
				body,
				std::vector<std::shared_ptr<ASTNode>>{ orelse },
				SourceLocation{ elif_token.start(), orelse->source_location().end });
		}

		// 'elif' named_expression ':' block [else_block]
		using pattern2 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ElifKeywordPattern>,
			NamedExpressionPattern,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern,
			ZeroOrOnePatternV2<ElseBlockPattern>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'elif' named_expression ':' block [else_block]");
			auto [elif_token, test, _, body, orelse] = *result;
			if (orelse.has_value()) {
				return std::make_shared<If>(test,
					body,
					*orelse,
					SourceLocation{ elif_token.start(), orelse->back()->source_location().end });
			} else {
				return std::make_shared<If>(test,
					body,
					std::vector<std::shared_ptr<ASTNode>>{},
					SourceLocation{ elif_token.start(), body.back()->source_location().end });
			}
		}

		return {};
	}
};

template<> struct traits<struct IfStatementPattern>
{
	using result_type = std::shared_ptr<If>;
};

struct IfStatementPattern : PatternV2<IfStatementPattern>
{
	using ResultType = typename traits<IfStatementPattern>::result_type;

	// if_stmt:
	//     | 'if' named_expression ':' block elif_stmt
	//     | 'if' named_expression ':' block [else_block]
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("if_stmt");

		// 'if' named_expression ':' block elif_stmt
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, IfKeywordPattern>,
			NamedExpressionPattern,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern,
			ElifStatementPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'if' named_expression ':' block elif_stmt");
			auto [if_token, test, _, body, orelse] = *result;
			return std::make_shared<If>(test,
				body,
				std::vector<std::shared_ptr<ASTNode>>{ orelse },
				SourceLocation{ if_token.start(), orelse->source_location().end });
		}

		// 'if' named_expression ':' block [else_block]
		using pattern2 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, IfKeywordPattern>,
			NamedExpressionPattern,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern,
			ZeroOrOnePatternV2<ElseBlockPattern>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'if' named_expression ':' block [else_block]");
			auto [if_token, test, _, body, orelse] = *result;
			if (orelse.has_value()) {
				return std::make_shared<If>(test,
					body,
					*orelse,
					SourceLocation{ if_token.start(), orelse->back()->source_location().end });
			} else {
				return std::make_shared<If>(test,
					body,
					std::vector<std::shared_ptr<ASTNode>>{},
					SourceLocation{ if_token.start(), body.back()->source_location().end });
			}
		}

		return {};
	}
};

template<> struct traits<struct ClassDefinitionRawPattern>
{
	using result_type = std::shared_ptr<ClassDefinition>;
};

struct ClassDefinitionRawPattern : PatternV2<ClassDefinitionRawPattern>
{
	using ResultType = typename traits<ClassDefinitionRawPattern>::result_type;

	// class_def_raw:
	//     | 'class' NAME ['(' [arguments] ')' ] ':' block
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("class_def_raw");

		// 'class' NAME ['(' [arguments] ')' ] ':' block
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ClassKeywordPattern>,
			SingleTokenPatternV2<Token::TokenType::NAME>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::LPAREN>,
				ZeroOrOnePatternV2<ArgumentsPattern>,
				SingleTokenPatternV2<Token::TokenType::RPAREN>>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'class' NAME ['(' [arguments] ')' ] ':' block");
			auto [class_token, class_name_token, arguments, _, body] = *result;
			std::vector<std::shared_ptr<ASTNode>> bases;
			std::vector<std::shared_ptr<Keyword>> keywords;
			if (arguments.has_value()) {
				auto [l, args, r] = *arguments;
				if (args.has_value()) {
					auto [bases_, keywords_] = *args;
					bases = bases_;
					keywords = keywords_;
				}
			}
			std::vector<std::shared_ptr<ASTNode>> decorator_list;
			std::string class_name{ class_name_token.token.start().pointer_to_program,
				class_name_token.token.end().pointer_to_program };

			return std::make_shared<ClassDefinition>(class_name,
				bases,
				keywords,
				body,
				decorator_list,
				SourceLocation{ class_token.start(), body.back()->source_location().end });
		}

		return {};
	}
};

template<> struct traits<struct ClassDefinitionPattern>
{
	using result_type = std::shared_ptr<ClassDefinition>;
};

struct ClassDefinitionPattern : PatternV2<ClassDefinitionPattern>
{
	using ResultType = typename traits<ClassDefinitionPattern>::result_type;

	// class_def:
	//     | decorators class_def_raw
	//     | class_def_raw
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("class_def");

		// decorators class_def_raw
		using pattern1 = PatternMatchV2<DecoratorsPattern, ClassDefinitionRawPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("decorators class_def_raw");
			auto [decorators, class_def] = *result;
			return std::make_shared<ClassDefinition>(class_def->name(),
				class_def->bases(),
				class_def->keywords(),
				class_def->body(),
				decorators.second,
				SourceLocation{ decorators.first.start(), class_def->source_location().end });
		}

		// class_def_raw
		using pattern2 = PatternMatchV2<ClassDefinitionRawPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("class_def_raw");
			auto [class_def] = *result;
			return class_def;
		}
		return {};
	}
};

template<> struct traits<struct ForStatementPattern>
{
	using result_type = std::shared_ptr<For>;
};

struct ForStatementPattern : PatternV2<ForStatementPattern>
{
	using ResultType = typename traits<ForStatementPattern>::result_type;

	// for_stmt:
	//     | 'for' star_targets 'in' ~ star_expressions ':' [TYPE_COMMENT] block [else_block]
	//     | ASYNC 'for' star_targets 'in' ~ star_expressions ':' [TYPE_COMMENT] block [else_block]
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("for_stmt");

		// 'for' star_targets 'in' ~ star_expressions ':' [TYPE_COMMENT] block [else_block]
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ForKeywordPattern>,
			StarTargetsPattern,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, InKeywordPattern>,
			StarExpressionsPattern,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern,
			ZeroOrOnePatternV2<ElseBlockPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'for' star_targets 'in' ~ star_expressions ':' [TYPE_COMMENT] block");
			auto [for_token, targets, in_token, iter, colon_token, body, orelse] = *result;
			std::string type_comment;
			return std::make_shared<For>(targets,
				iter,
				body,
				orelse.value_or(std::vector<std::shared_ptr<ASTNode>>{}),
				type_comment,
				SourceLocation{ for_token.start(),
					orelse.has_value() ? orelse->back()->source_location().end
									   : body.back()->source_location().end });
		}

		// ASYNC 'for' star_targets 'in' ~ star_expressions ':' [TYPE_COMMENT] block [else_block]
		using pattern2 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, AsyncKeywordPattern>,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ForKeywordPattern>,
			StarTargetsPattern,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, InKeywordPattern>,
			StarExpressionsPattern,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern,
			ZeroOrOnePatternV2<ElseBlockPattern>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'for' star_targets 'in' ~ star_expressions ':' [TYPE_COMMENT] block");
			auto [async_token, for_token, targets, in_token, iter, colon_token, body, orelse] =
				*result;
			TODO_NO_FAIL();
		}

		return {};
	}
};

template<> struct traits<struct ExceptBlockPattern>
{
	using result_type = std::shared_ptr<ExceptHandler>;
};

struct ExceptBlockPattern : PatternV2<ExceptBlockPattern>
{
	using ResultType = typename traits<ExceptBlockPattern>::result_type;

	// except_block:
	//     | 'except' expression ['as' NAME ] ':' block
	//     | 'except' ':' block
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("except_block");

		// 'except' expression ['as' NAME ] ':' block
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ExceptKeywordPattern>,
			ExpressionPattern,
			ZeroOrOnePatternV2<
				AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, AsKeywordPattern>,
				SingleTokenPatternV2<Token::TokenType::NAME>>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'except' expression ['as' NAME ] ':' block");

			auto [except_token, type, as_name, _, body] = *result;

			auto name = [as_name_ = as_name]() -> std::string {
				if (as_name_.has_value()) {
					auto [_, name_token] = *as_name_;
					return { name_token.token.start().pointer_to_program,
						name_token.token.end().pointer_to_program };
				} else {
					return "";
				}
			}();

			return std::make_shared<ExceptHandler>(type,
				name,
				body,
				SourceLocation{ except_token.start(), body.back()->source_location().end });
		}

		// 'except' ':' block
		using pattern2 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, ExceptKeywordPattern>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'except' ':' block");

			auto [except_token, _, body] = *result;
			std::shared_ptr<ASTNode> type;
			std::string name;
			return std::make_shared<ExceptHandler>(type,
				name,
				body,
				SourceLocation{ except_token.start(), body.back()->source_location().end });
		}

		return {};
	}
};

template<> struct traits<struct FinallyBlockPattern>
{
	using result_type = typename traits<BlockPattern>::result_type;
};

struct FinallyBlockPattern : PatternV2<FinallyBlockPattern>
{
	using ResultType = typename traits<FinallyBlockPattern>::result_type;

	// finally_block:
	//     | 'finally' ':' block
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("finally_block");
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, FinallyKeywordPattern>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'finally' ':' block");
			auto [finally_token, _, block] = *result;
			return block;
		}
		return {};
	}
};

template<> struct traits<struct TryStatementPattern>
{
	using result_type = std::shared_ptr<Try>;
};

struct TryStatementPattern : PatternV2<TryStatementPattern>
{
	using ResultType = typename traits<TryStatementPattern>::result_type;

	// try_stmt:
	// 		| 'try' ':' block finally_block
	// 		| 'try' ':' block except_block+ [else_block] [finally_block]
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("try_stmt");

		// 'try' ':' block finally_block
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, TryKeywordPattern>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern,
			FinallyBlockPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'try' ':' block finally_block");

			auto [try_token, _, body, finally] = *result;

			std::vector<std::shared_ptr<ExceptHandler>> handlers;
			std::vector<std::shared_ptr<ASTNode>> orelse;

			return std::make_shared<Try>(body,
				handlers,
				orelse,
				finally,
				SourceLocation{ try_token.start(), finally.back()->source_location().end });
		}

		// 'try' ':' block except_block+ [else_block] [finally_block]
		using pattern2 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, TryKeywordPattern>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern,
			OneOrMorePatternV2<ExceptBlockPattern>,
			ZeroOrOnePatternV2<ElseBlockPattern>,
			ZeroOrOnePatternV2<FinallyBlockPattern>>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'try' ':' block except_block+ [else_block] [finally_block]");

			auto [try_token, _, body, handlers, orelse, finally] = *result;
			const auto end = p.lexer().peek_token(p.token_position() - 1);
			return std::make_shared<Try>(body,
				handlers,
				orelse.value_or(std::vector<std::shared_ptr<ASTNode>>{}),
				finally.value_or(std::vector<std::shared_ptr<ASTNode>>{}),
				SourceLocation{ try_token.start(), end->end() });
		}

		return {};
	}
};

template<> struct traits<struct WhileStatementPattern>
{
	using result_type = std::shared_ptr<While>;
};

struct WhileStatementPattern : PatternV2<WhileStatementPattern>
{
	using ResultType = typename traits<WhileStatementPattern>::result_type;

	// while_stmt:
	//     | 'while' named_expression ':' block [else_block]
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("while_stmt");

		// 'while' named_expression ':' block
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, WhileKeywordPattern>,
			NamedExpressionPattern,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern,
			ZeroOrOnePatternV2<ElseBlockPattern>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'while' named_expression ':' block [else_block]");
			auto [while_token, test, _, body, orelse] = *result;

			const auto end_token = p.lexer().peek_token(p.token_position() - 1);
			return std::make_shared<While>(test,
				body,
				orelse.value_or(std::vector<std::shared_ptr<ASTNode>>{}),
				SourceLocation{ while_token.start(), end_token->end() });
		}

		return {};
	}
};

template<> struct traits<struct WithItemPattern>
{
	using result_type = std::shared_ptr<WithItem>;
};

struct WithItemPattern : PatternV2<WithItemPattern>
{
	using ResultType = typename traits<WithItemPattern>::result_type;

	// with_item:
	//     | expression 'as' star_target &(',' | ')' | ':')
	//     | expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("with_item");

		using pattern1 = PatternMatchV2<ExpressionPattern,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, AsKeywordPattern>,
			StarTargetPattern,
			LookAheadV2<OrPatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>,
				SingleTokenPatternV2<Token::TokenType::RPAREN>,
				SingleTokenPatternV2<Token::TokenType::COLON>>>>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("expression 'as' star_target &(',' | ')' | ':')");
			auto [context_expr, as_token, var, _] = *result;
			return std::make_shared<WithItem>(context_expr,
				var,
				SourceLocation{
					context_expr->source_location().start, var->source_location().end });
		}

		using pattern2 = PatternMatchV2<ExpressionPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("expression");
			auto [context_expr] = *result;
			return std::make_shared<WithItem>(context_expr,
				nullptr,
				SourceLocation{
					context_expr->source_location().start, context_expr->source_location().end });
		}

		return {};
	}
};

template<> struct traits<struct WithStatementPattern>
{
	using result_type = std::shared_ptr<With>;
};

struct WithStatementPattern : PatternV2<WithStatementPattern>
{
	using ResultType = typename traits<WithStatementPattern>::result_type;

	// with_stmt:
	//     | 'with' '(' ','.with_item+ ','? ')' ':' block
	//     | 'with' ','.with_item+ ':' [TYPE_COMMENT] block
	//     | ASYNC 'with' '(' ','.with_item+ ','? ')' ':' block
	//     | ASYNC 'with' ','.with_item+ ':' [TYPE_COMMENT] block
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		DEBUG_LOG("with_stmt");

		// 'with' '(' ','.with_item+ ','? ')' ':' block
		using pattern1 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, WithKeywordPattern>,
			SingleTokenPatternV2<Token::TokenType::LPAREN>,
			ApplyInBetweenPatternV2<WithItemPattern, SingleTokenPatternV2<Token::TokenType::COMMA>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("'with' '(' ','.with_item+ ','? ')' ':' block");

			auto [with_token, lp, with_items, trailing_comma, rp, _, body] = *result;

			return std::make_shared<With>(with_items,
				body,
				"",
				SourceLocation{ with_token.start(), body.back()->source_location().end });
		}

		// 'with' ','.with_item+ ':' [TYPE_COMMENT] block
		using pattern2 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, WithKeywordPattern>,
			ApplyInBetweenPatternV2<WithItemPattern, SingleTokenPatternV2<Token::TokenType::COMMA>>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("'with' ','.with_item+ ':' [TYPE_COMMENT] block");

			auto [with_token, with_items, _, body] = *result;

			return std::make_shared<With>(with_items,
				body,
				"",
				SourceLocation{ with_token.start(), body.back()->source_location().end });
		}

		// ASYNC 'with' '(' ','.with_item+ ','? ')' ':' block
		using pattern3 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, AsyncKeywordPattern>,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, WithKeywordPattern>,
			SingleTokenPatternV2<Token::TokenType::LPAREN>,
			OneOrMorePatternV2<ApplyInBetweenPatternV2<WithItemPattern,
				SingleTokenPatternV2<Token::TokenType::COMMA>>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>,
			SingleTokenPatternV2<Token::TokenType::RPAREN>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("ASYNC 'with' '(' ','.with_item+ ','? ')' ':' block");
			(void)result;
			TODO_NO_FAIL();
		}

		// ASYNC 'with' ','.with_item+ ':' [TYPE_COMMENT] block
		using pattern4 = PatternMatchV2<
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, AsyncKeywordPattern>,
			AndLiteralV2<SingleTokenPatternV2<Token::TokenType::NAME>, WithKeywordPattern>,
			ApplyInBetweenPatternV2<WithItemPattern, SingleTokenPatternV2<Token::TokenType::COMMA>>,
			SingleTokenPatternV2<Token::TokenType::COLON>,
			BlockPattern>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("ASYNC 'with' ','.with_item+ ':' [TYPE_COMMENT] block");
			(void)result;
			TODO_NO_FAIL();
		}

		return {};
	}
};

template<> struct traits<struct CompoundStatementPattern>
{
	using result_type = std::shared_ptr<ASTNode>;
};

struct CompoundStatementPattern : PatternV2<CompoundStatementPattern>
{
	using ResultType = typename traits<CompoundStatementPattern>::result_type;

	// compound_stmt:
	//     | function_def
	//     | if_stmt
	//     | class_def
	//     | with_stmt
	//     | for_stmt
	//     | try_stmt
	//     | while_stmt
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// function_def
		using pattern1 = PatternMatchV2<FunctionDefinitionStatementPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("function_def");
			auto [function_def] = *result;
			return function_def;
		}

		// if_stmt
		using pattern2 = PatternMatchV2<IfStatementPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("if_stmt");
			auto [if_stmt] = *result;
			return if_stmt;
		}

		// class_def
		using pattern3 = PatternMatchV2<ClassDefinitionPattern>;
		if (auto result = pattern3::match(p)) {
			DEBUG_LOG("class_def");
			auto [class_def] = *result;
			return class_def;
		}

		// with_stmt
		using pattern4 = PatternMatchV2<WithStatementPattern>;
		if (auto result = pattern4::match(p)) {
			DEBUG_LOG("with_stmt");
			auto [with_stmt] = *result;
			return with_stmt;
		}

		// for_stmt
		using pattern5 = PatternMatchV2<ForStatementPattern>;
		if (auto result = pattern5::match(p)) {
			DEBUG_LOG("for_stmt");
			auto [for_stmt] = *result;
			return for_stmt;
		}

		// try_stmt
		using pattern6 = PatternMatchV2<TryStatementPattern>;
		if (auto result = pattern6::match(p)) {
			DEBUG_LOG("try_stmt");
			auto [try_stmt] = *result;
			return try_stmt;
		}

		// while_stmt
		using pattern7 = PatternMatchV2<WhileStatementPattern>;
		if (auto result = pattern7::match(p)) {
			DEBUG_LOG("while_stmt");
			auto [while_stmt] = *result;
			return while_stmt;
		}
		return {};
	}
};

template<> struct traits<struct StatementPattern>
{
	using result_type = std::vector<std::shared_ptr<ASTNode>>;
};


struct StatementPattern : PatternV2<StatementPattern>
{
	using ResultType = typename traits<StatementPattern>::result_type;

	// statement: compound_stmt
	// 			  | simple_stmt
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		// compound_stmt
		using pattern1 = PatternMatchV2<CompoundStatementPattern>;
		if (auto result = pattern1::match(p)) {
			DEBUG_LOG("compound_stmt");
			auto [compound_stmt] = *result;
			return { { compound_stmt } };
		}

		// simple_stmt
		using pattern2 = PatternMatchV2<SimpleStatementPattern>;
		if (auto result = pattern2::match(p)) {
			DEBUG_LOG("simple_stmt");
			auto [simple_stmt] = *result;
			return simple_stmt;
		}
		return {};
	}
};

struct StatementsPattern : PatternV2<StatementsPattern>
{
	using ResultType = typename traits<StatementsPattern>::result_type;

	// statements: statement+
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<StatementPattern>;
		ResultType statements;
		while (auto result = pattern1::match(p)) {
			// p.m_cache.clear();
			auto [statement] = *result;
			statements.insert(statements.end(), statement.begin(), statement.end());
		}
		if (statements.empty()) { return std::nullopt; }
		return statements;
	}
};

template<> struct traits<struct ExpressionsPattern>
{
	using result_type = std::vector<std::shared_ptr<ASTNode>>;
};

struct ExpressionsPattern : PatternV2<ExpressionsPattern>
{
	using ResultType = typename traits<ExpressionsPattern>::result_type;

	// expressions: expression ((',' expression))+ ','?
	// 				| expression ','
	// 				| expression
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<ApplyInBetweenPatternV2<ExpressionPattern,
											SingleTokenPatternV2<Token::TokenType::COMMA>>,
			ZeroOrOnePatternV2<SingleTokenPatternV2<Token::TokenType::COMMA>>>;
		if (auto result = pattern1::match(p)) {
			auto [expressions, _] = *result;
			return expressions;
		}

		using pattern2 =
			PatternMatchV2<ExpressionPattern, SingleTokenPatternV2<Token::TokenType::COMMA>>;
		if (auto result = pattern2::match(p)) {
			auto [expression, _] = *result;
			return { { expression } };
		}
		using pattern3 = PatternMatchV2<ExpressionPattern>;
		if (auto result = pattern3::match(p)) {
			auto [expression] = *result;
			return { { expression } };
		}

		return {};
	}
};

template<> struct traits<struct FilePattern>
{
	using result_type = std::shared_ptr<Module>;
};

struct FilePattern : PatternV2<FilePattern>
{
	using ResultType = typename traits<FilePattern>::result_type;

	// file: [statements] ENDMARKER
	static std::optional<ResultType> matches_impl(Parser &p)
	{
		using pattern1 = PatternMatchV2<ZeroOrOnePatternV2<StatementsPattern>,
			SingleTokenPatternV2<Token::TokenType::ENDMARKER>>;
		if (auto result = pattern1::match(p)) {
			auto [statements, _] = *result;
			if (statements.has_value()) {
				for (auto &&statement : *statements) { p.module()->emplace(std::move(statement)); }
			}
			return p.module();
		}
		size_t idx = 0;
		auto t = *p.lexer().peek_token(idx);
		auto begin = t.start().pointer_to_program;
		auto end = t.end().pointer_to_program;
		const size_t row = t.start().row;
		while (row == t.start().row) {
			end = t.end().pointer_to_program;
			idx++;
			t = *p.lexer().peek_token(idx);
		}
		std::string line{ begin, end };
		spdlog::error("Syntax error on line {}: '{}'", row + 1, line);
		// PARSER_ERROR();
		return {};
	}
};

namespace parser {
void Parser::parse()
{
	auto result = PatternMatchV2<FilePattern>::match(*this);
	if (result) {
		auto [module] = *result;
		m_module = std::move(module);
		m_module->print_node("");
	}
	DEBUG_LOG("Parser return code: {}", result.has_value());
}

PyResult<std::shared_ptr<ast::Module>> Parser::parse_expression()
{
	// eval: expressions NEWLINE* $
	auto result = PatternMatchV2<ExpressionsPattern,
		ZeroOrMorePatternV2<SingleTokenPatternV2<Token::TokenType::NEWLINE>>,
		SingleTokenPatternV2<Token::TokenType::ENDMARKER>>::match(*this);
	if (result.has_value()) {
		auto expressions = std::get<0>(*result);
		if (expressions.size() == 1) {
			m_module->emplace(std::make_shared<Return>(
				expressions.back(), expressions.back()->source_location()));
		} else {
			auto result = std::make_shared<Tuple>(expressions,
				ContextType::LOAD,
				SourceLocation{ .start = expressions.front()->source_location().start,
					.end = expressions.back()->source_location().end });
			m_module->emplace(std::make_shared<Return>(result, result->source_location()));
		}
		m_module->print_node("");
		return Ok(m_module);
	}
	return Err(syntax_error(m_lexer.program()));
}

PyResult<std::shared_ptr<ASTNode>> Parser::parse_fstring()
{
	// fstring: star_expressions
	auto result = PatternMatchV2<StarExpressionPattern>::match(*this);
	if (result.has_value()) {
		auto [expressions] = *result;
		return Ok(expressions);
	}
	return Err(syntax_error(m_lexer.program()));
}

}// namespace parser
