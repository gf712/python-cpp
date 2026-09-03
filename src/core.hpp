#pragma once

// Diagnostics macros and the two empty base classes, with no libstdc++ and no
// spdlog behind them.
//
// This header exists to be safe inside a module's global module fragment.
// Macros are never exported by a module, so any module unit that uses TODO() or
// ASSERT() has to #include them textually - and utilities.hpp, which used to be
// that include, drags in 272 libstdc++ headers through spdlog. Those headers
// then land in the module's BMI and collide with the same headers #included by
// its consumers ("redefinition of ...", "conflicting declaration of template
// ...").  Routing the macros through an out-of-line failure function keeps the
// formatting - and therefore spdlog - in core.cpp where it costs nothing.

namespace detail {
// Defined in core.cpp; reports through spdlog and aborts.
[[noreturn]] void assertion_failed(const char *what, const char *file, int line);

// Logging sinks. Module units cannot #include spdlog in their global module
// fragment - its libstdc++ headers land in the BMI and collide with the
// `import std;` the module interfaces use - so they format with std::format and
// pass the finished string through here.
void log_debug(const char *message);
void log_trace(const char *message);
void log_error(const char *message);
bool log_debug_enabled();
}// namespace detail

#define TODO() ::detail::assertion_failed("Not implemented", __FILE__, __LINE__)

#define ASSERT(condition)                                                                   \
	do {                                                                                    \
		if (!(condition)) {                                                                 \
			::detail::assertion_failed("Assertion failed " #condition, __FILE__, __LINE__); \
		}                                                                                   \
	} while (0)

#define ASSERT_NOT_REACHED() \
	::detail::assertion_failed("Reached unexpected line", __FILE__, __LINE__)

// Pure templates, no spdlog behind them - safe in a module GMF.
template<class... Ts> struct overloaded : Ts...
{
	using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

using Register = __UINT8_TYPE__;

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
