#pragma once

#include "EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include <map>
#include <set>
#include <variant>

namespace codegen {

// Represents an output from an operation that doesn't have explicit results
// but forwards a value to a successor block (e.g., ForIter forwards loop variable)
using ForwardedOutput = std::pair<mlir::Operation *, size_t>;

// Comparator for ValueMapping
struct ValueMappingComparator
{
	bool operator()(const std::variant<mlir::Value, ForwardedOutput> &lhs,
		const std::variant<mlir::Value, ForwardedOutput> &rhs) const
	{
		if (rhs.valueless_by_exception()) {
			return false;
		} else if (lhs.valueless_by_exception()) {
			return true;
		} else if (lhs.index() < rhs.index()) {
			return true;
		} else if (lhs.index() > rhs.index()) {
			return false;
		}
		if (std::holds_alternative<mlir::Value>(lhs)) {
			return std::get<mlir::Value>(lhs).getImpl() < std::get<mlir::Value>(rhs).getImpl();
		}
		return std::get<ForwardedOutput>(lhs) < std::get<ForwardedOutput>(rhs);
	}
};

// Map from MLIR values to some type T (used for register assignments, etc.)
template<typename ValueT>
using ValueMapping = std::map<std::variant<mlir::Value, ForwardedOutput>, ValueT, ValueMappingComparator>;

// Comparator for BlockArgument sets
inline constexpr auto block_arg_comparator = [](const mlir::BlockArgument &lhs,
												 const mlir::BlockArgument &rhs) {
	return static_cast<mlir::Value>(lhs).getImpl() < static_cast<mlir::Value>(rhs).getImpl();
};

// Helper functions to check operation types
inline bool is_function_call(mlir::Value value)
{
	if (!value.getDefiningOp()) return false;

	auto *op = value.getDefiningOp();
	return llvm::isa<mlir::emitpybytecode::FunctionCallOp>(op)
		   || llvm::isa<mlir::emitpybytecode::FunctionCallExOp>(op)
		   || llvm::isa<mlir::emitpybytecode::FunctionCallWithKeywordsOp>(op);
}

// Returns true if the operation clobbers register 0
// (function calls, with-except-start, yield, yield-from all write results to r0)
inline bool clobbers_r0(mlir::Value value)
{
	if (!value.getDefiningOp()) return false;

	auto *op = value.getDefiningOp();
	return is_function_call(value) || llvm::isa<mlir::emitpybytecode::WithExceptStart>(op)
		   || llvm::isa<mlir::emitpybytecode::Yield>(op)
		   || llvm::isa<mlir::emitpybytecode::YieldFrom>(op);
}

// Helper to sort blocks in dominator order
inline std::vector<mlir::Block *> sortBlocks(mlir::Region &region)
{
	auto result = mlir::getBlocksSortedByDominance(region);
	return { result.begin(), result.end() };
}

}// namespace codegen
