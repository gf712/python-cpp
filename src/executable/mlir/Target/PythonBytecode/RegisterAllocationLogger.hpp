#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

namespace codegen {

using ForwardedOutput = std::pair<mlir::Operation *, size_t>;

}

template<> struct fmt::formatter<mlir::Value, char>
{
	template<class ParseContext> constexpr ParseContext::iterator parse(ParseContext &ctx)
	{
		return ctx.begin();
	}

	template<class FmtContext> FmtContext::iterator format(mlir::Value value, FmtContext &ctx) const
	{
		std::string result;
		llvm::raw_string_ostream os(result);
		os << "Value@" << value.getImpl() << "[";
		value.print(os);
		os << "]";
		return fmt::format_to(ctx.out(), "{}", result);
	}
};

template<> struct fmt::formatter<codegen::ForwardedOutput, char>
{
	template<class ParseContext> constexpr ParseContext::iterator parse(ParseContext &ctx)
	{
		return ctx.begin();
	}

	template<class FmtContext>
	FmtContext::iterator format(codegen::ForwardedOutput output, FmtContext &ctx) const
	{
		std::string result;
		llvm::raw_string_ostream os(result);
		os << "ForwardedOutput@" << static_cast<void *>(output.first) << "[";
		output.first->print(os);
		os << ", result_idx=" << output.second << "]";
		return fmt::format_to(ctx.out(), "{}", result);
	}
};


template<> struct fmt::formatter<std::variant<mlir::Value, codegen::ForwardedOutput>, char>
{

	template<class ParseContext> constexpr ParseContext::iterator parse(ParseContext &ctx)
	{
		return ctx.begin();
	}

	template<class FmtContext>
	FmtContext::iterator format(std::variant<mlir::Value, codegen::ForwardedOutput> value,
		FmtContext &ctx) const
	{
		if (std::holds_alternative<mlir::Value>(value)) {
			return fmt::format_to(ctx.out(), "{}", std::get<mlir::Value>(value));
		} else {
			return fmt::format_to(ctx.out(), "{}", std::get<codegen::ForwardedOutput>(value));
		}
	}
};


namespace codegen {

// Parse log level string to spdlog level enum
inline spdlog::level::level_enum parse_log_level(const char *level_str,
	spdlog::level::level_enum default_level)
{
	if (!level_str) { return default_level; }

	std::string level(level_str);
	// Convert to lowercase for case-insensitive comparison
	std::transform(
		level.begin(), level.end(), level.begin(), [](unsigned char c) { return std::tolower(c); });

	if (level == "trace") {
		return spdlog::level::trace;
	} else if (level == "debug") {
		return spdlog::level::debug;
	} else if (level == "info") {
		return spdlog::level::info;
	} else if (level == "warn" || level == "warning") {
		return spdlog::level::warn;
	} else if (level == "error" || level == "err") {
		return spdlog::level::err;
	} else if (level == "critical") {
		return spdlog::level::critical;
	} else if (level == "off") {
		return spdlog::level::off;
	}

	return default_level;
}

// Get the logger for register allocation
inline std::shared_ptr<spdlog::logger> get_regalloc_logger()
{
	static auto logger = []() {
		auto l = spdlog::get("regalloc");
		if (!l) { l = spdlog::stdout_color_mt("regalloc"); }
		auto log_level = parse_log_level(std::getenv("LOG_REGALLOC"), spdlog::level::err);
		l->set_level(log_level);
		return l;
	}();
	return logger;
}

}// namespace codegen
