#pragma once

// fmt formatting for ast::SourceLocation. Kept out of the py.ast module for the
// same reason as PositionFormatter.hpp: spdlog/fmt/fmt.h drags 231 libstdc++
// headers into the BMI, which then collide with consumers' own #includes.


template<> struct std::formatter<SourceLocation>
{
	constexpr auto parse(std::format_parse_context &ctx) { return ctx.end(); }

	template<typename FormatContext> auto format(const SourceLocation &sc, FormatContext &ctx) const
	{
		return std::format_to(ctx.out(), "[{}-{}]", sc.start, sc.end);
	}
};
