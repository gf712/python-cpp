#pragma once

// fmt formatting for lexer::Position.
//
// Kept out of the py.lexer module on purpose: spdlog/fmt/fmt.h pulls in 231
// libstdc++ headers, and anything in a module's global module fragment ends up
// in its BMI, where it collides with the same headers #included by consumers.
// Only the few places that log a Position need this, so they include it here.

#include <format>

template<> struct std::formatter<Position>
{
	constexpr auto parse(std::format_parse_context &ctx) { return ctx.end(); }

	template<typename FormatContext> auto format(const Position &pos, FormatContext &ctx) const
	{
		return std::format_to(ctx.out(), "{}:{}", pos.row + 1, pos.column);
	}
};
