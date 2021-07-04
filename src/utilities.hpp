#pragma once

#include "spdlog/spdlog.h"

#define TODO()                                                  \
	spdlog::error("Not implemented {}:{}", __FILE__, __LINE__); \
	std::abort();

#define ASSERT(condition)                                                           \
	if (!(condition)) {                                                             \
		spdlog::error("Assertion failed {} {}:{}", #condition, __FILE__, __LINE__); \
		std::abort();                                                               \
	}

#define ASSERT_NOT_REACHED() __builtin_unreachable();

template<class... Ts> struct overloaded : Ts...
{
	using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;