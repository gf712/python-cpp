#pragma once

#include "spdlog/spdlog.h"
#include <bit>

#define TODO()                                                  \
	spdlog::error("Not implemented {}:{}", __FILE__, __LINE__); \
	std::abort();

#define ASSERT(condition)                                                           \
	if (!(condition)) {                                                             \
		spdlog::error("Assertion failed {} {}:{}", #condition, __FILE__, __LINE__); \
		std::abort();                                                               \
	}

#define ASSERT_NOT_REACHED()                                            \
	spdlog::error("Reached unexpected line {}:{}", __FILE__, __LINE__); \
	__builtin_unreachable();

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

#if !defined(STL_SUPPORTS_BIT_CAST)
template<class To, class From>
typename std::enable_if_t<
	sizeof(To) == sizeof(From)
		&& std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>,
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
{
	return std::bit_cast<To>(from);
}
#endif