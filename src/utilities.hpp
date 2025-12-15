#pragma once

#include "spdlog/spdlog.h"
#include <bit>

#define TODO()                                                      \
	do {                                                            \
		spdlog::error("Not implemented {}:{}", __FILE__, __LINE__); \
		std::abort();                                               \
	} while (0)

#define ASSERT(condition)                                                               \
	do {                                                                                \
		if (!(condition)) {                                                             \
			spdlog::error("Assertion failed {} {}:{}", #condition, __FILE__, __LINE__); \
			std::abort();                                                               \
		}                                                                               \
	} while (0)

#define ASSERT_NOT_REACHED()                                                \
	do {                                                                    \
		spdlog::error("Reached unexpected line {}:{}", __FILE__, __LINE__); \
		__builtin_unreachable();                                            \
		std::abort();                                                       \
	} while (0)
template<class... Ts> struct overloaded : Ts...
{
	using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

struct NonCopyable
{
	NonCopyable() = default;
	NonCopyable(const NonCopyable &) = delete;
	NonCopyable &operator=(const NonCopyable &) = delete;
};

struct NonMoveable
{
	NonMoveable() = default;
	NonMoveable(NonMoveable &&) = delete;
	NonMoveable &operator=(NonMoveable &&) = delete;
};


namespace detail {
template<class T> struct member_pointer_helper : std::false_type
{
};

template<class T, class U> struct member_pointer_helper<T U::*> : std::true_type
{
	using type = T;
};
}// namespace detail

template<class T>
struct member_pointer : detail::member_pointer_helper<typename std::remove_cv<T>::type>
{
};


#if !defined(STL_SUPPORTS_BIT_CAST)
template<class To, class From>
typename std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From>
							  && std::is_trivially_copyable_v<To>,
	To>
	// constexpr support needs compiler magic
	bit_cast(const From &src) noexcept
{
	static_assert(std::is_trivially_constructible_v<To>,
		"This implementation additionally requires destination type to be trivially constructible");

	To dst;
	std::memcpy(&dst, &src, sizeof(To));
	return dst;
}
#else
template<class To, class From> constexpr To bit_cast(const From &from) noexcept
{ return std::bit_cast<To>(from); }
#endif
