#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <memory>
#include <sstream>
#include <string>
#include <variant>

namespace codegen {

using ForwardedOutput = std::pair<mlir::Operation *, size_t>;

// Helper functions to convert MLIR values to strings for logging
inline std::string to_string(const mlir::Value &value)
{
	std::string result;
	llvm::raw_string_ostream os(result);
	os << "Value@" << value.getImpl() << "[";
	value.print(os);
	os << "]";
	return result;
}

inline std::string to_string(const ForwardedOutput &output)
{
	std::string result;
	llvm::raw_string_ostream os(result);
	os << "ForwardedOutput@" << static_cast<void *>(output.first) << "[";
	output.first->print(os);
	os << ", result_idx=" << output.second << "]";
	return result;
}

inline std::string to_string(const std::variant<mlir::Value, ForwardedOutput> &value)
{
	if (std::holds_alternative<mlir::Value>(value)) {
		return to_string(std::get<mlir::Value>(value));
	} else {
		return to_string(std::get<ForwardedOutput>(value));
	}
}

// Get the logger for register allocation
inline std::shared_ptr<spdlog::logger> get_regalloc_logger()
{
	static auto logger = []() {
		auto l = spdlog::get("regalloc");
		if (!l) {
			l = spdlog::stdout_color_mt("regalloc");
			// Set default level to warning (can be changed via spdlog::set_level)
			l->set_level(spdlog::level::warn);
		}
		return l;
	}();
	return logger;
}

}// namespace codegen
