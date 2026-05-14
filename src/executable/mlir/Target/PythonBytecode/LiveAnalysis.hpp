#pragma once

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "RegisterAllocationLogger.hpp"
#include "RegisterAllocationTypes.hpp"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "utilities.hpp"

#include <algorithm>
#include <map>
#include <ranges>
#include <set>
#include <span>
#include <tuple>
#include <vector>

namespace codegen {

/**
 * LiveAnalysis determines which values are alive at each point in the program.
 * This is the first step in register allocation.
 *
 * The block-level fixed-point dataflow is delegated to mlir::Liveness — it
 * already implements the standard LiveOut[B] = ∪ LiveIn[S] for S in succ(B),
 * LiveIn[B] = Use[B] ∪ (LiveOut[B] - Def[B]) algorithm and handles loops /
 * back-edges. What's project-specific stays here:
 *
 *   1. ForwardedOutput for FOR_ITER. The body block's first argument is the
 *      loop value produced by FOR_ITER's terminator, which MLIR can't model
 *      natively. We track it as a synthetic Value-like entity.
 *
 *   2. alive_at_timestep — a linearised per-operation list of "what's alive
 *      here". mlir::Liveness only exposes per-block / per-value queries; the
 *      register allocator wants the linearised view, so we materialise it on
 *      top of mlir::Liveness's block-level results.
 *
 *   3. block_input_mappings — for each value, the set of block arguments it
 *      could flow into via a CFG edge. Computed from the terminator operands.
 */
class LiveAnalysis
{
  public:
	// Represents a block argument and all the values that can flow into it
	using BlockArgumentInputs =
		std::tuple<mlir::BlockArgument, std::vector<std::variant<mlir::Value, ForwardedOutput>>>;

	// Values alive at a single timestep (operation)
	using AliveAtTimestepT =
		std::vector<std::variant<mlir::Value, ForwardedOutput, BlockArgumentInputs>>;

	// Set of values (for dataflow sets)
	using ValueSet = std::set<std::variant<mlir::Value, ForwardedOutput>, ValueMappingComparator>;

	// Values alive at each timestep
	std::vector<AliveAtTimestepT> alive_at_timestep;

	// Maps values to block arguments they flow into
	ValueMapping<std::set<mlir::BlockArgument, decltype(block_arg_comparator)>>
		block_input_mappings;

	/**
	 * Analyze the function to determine liveness information.
	 */
	void analyse(mlir::func::FuncOp &fn)
	{
		auto logger = get_regalloc_logger();
		logger->info("Starting live analysis for function: {}", fn.getName().str());

		auto &region = fn.getRegion();
		auto sorted_blocks = sortBlocks(region);

		// Collect block operations and the project-specific bits MLIR doesn't
		// model natively (ForwardedOutputs from FOR_ITER, value→block-arg
		// edge mapping). Use/def computation is no longer needed here —
		// mlir::Liveness owns the dataflow.
		std::map<mlir::Block *, BlockInfo> block_info;
		std::vector<std::pair<std::variant<mlir::Value, ForwardedOutput>, mlir::BlockArgument>>
			block_parameters_to_args;

		for (auto *block : sorted_blocks) {
			build_block_info(block, block_info[block], block_parameters_to_args, logger);
		}

		// Block-level live-in / live-out via the upstream analysis.
		mlir::Liveness liveness(fn);

		// Materialise alive_at_timestep on top of mlir::Liveness's block
		// results, injecting ForwardedOutputs into the live sets as needed.
		std::map<mlir::Block *, std::pair<size_t, size_t>> blocks_span;
		build_timesteps(sorted_blocks, block_info, liveness, blocks_span);

		// Propagate block argument inputs through the liveness information
		propagate_block_arguments(block_parameters_to_args, blocks_span);

		// Resolve transitive block argument mappings
		resolve_block_argument_chains();

		logger->info("Live analysis complete. Tracked {} timesteps", alive_at_timestep.size());
	}

  private:
	/**
	 * Per-block ancillary state: the operation list (in topological order,
	 * needed because the linearised timesteps must match the order ops are
	 * walked by the bytecode emitter) plus the ForwardedOutputs created by
	 * this block's terminator (FOR_ITER's synthetic loop-var value).
	 */
	struct BlockInfo
	{
		// Operations in this block (in order)
		std::vector<mlir::Operation *> operations;

		// ForwardedOutputs created by terminators in this block
		std::vector<ForwardedOutput> forwarded_outputs;
	};

	/**
	 * Collect operations and project-specific terminator metadata for a block.
	 * Block-level live-in/out is computed separately by mlir::Liveness.
	 */
	void build_block_info(mlir::Block *block,
		BlockInfo &info,
		std::vector<std::pair<std::variant<mlir::Value, ForwardedOutput>, mlir::BlockArgument>>
			&block_parameters_to_args,
		std::shared_ptr<spdlog::logger> &logger)
	{
		// Sort operations topologically within the block
		if (!sortTopologically(block)) {
			logger->error("Failed to sort block topologically");
			std::abort();
		}

		for (auto &op : block->getOperations()) { info.operations.push_back(&op); }

		// Handle terminators specially
		if (auto *terminator = block->getTerminator()) {
			handle_terminator(terminator, info, block_parameters_to_args, logger);
		}
	}

	/**
	 * Handle terminator to extract ForwardedOutputs and block argument mappings
	 */
	void handle_terminator(mlir::Operation *terminator,
		BlockInfo &info,
		std::vector<std::pair<std::variant<mlir::Value, ForwardedOutput>, mlir::BlockArgument>>
			&block_parameters_to_args,
		std::shared_ptr<spdlog::logger> &logger)
	{
		if (auto branch = mlir::dyn_cast<mlir::emitpybytecode::JumpIfFalse>(terminator)) {
			handle_conditional_branch(branch, block_parameters_to_args);
		} else if (auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(terminator)) {
			handle_unconditional_branch(branch, block_parameters_to_args);
		} else if (auto for_iter = mlir::dyn_cast<mlir::emitpybytecode::ForIter>(terminator)) {
			handle_for_iter(for_iter, info, block_parameters_to_args, logger);
		}
	}

	void handle_conditional_branch(mlir::emitpybytecode::JumpIfFalse branch,
		std::vector<std::pair<std::variant<mlir::Value, ForwardedOutput>, mlir::BlockArgument>>
			&block_parameters_to_args)
	{
		auto *true_block = branch.getTrueDest();
		for (const auto &[p, arg] :
			llvm::zip(branch.getTrueDestOperands(), true_block->getArguments())) {
			block_parameters_to_args.emplace_back(p, arg);
		}

		auto *false_block = branch.getFalseDest();
		for (const auto &[p, arg] :
			llvm::zip(branch.getFalseDestOperands(), false_block->getArguments())) {
			block_parameters_to_args.emplace_back(p, arg);
		}
	}

	void handle_unconditional_branch(mlir::cf::BranchOp branch,
		std::vector<std::pair<std::variant<mlir::Value, ForwardedOutput>, mlir::BlockArgument>>
			&block_parameters_to_args)
	{
		auto *jmp_block = branch.getDest();
		for (const auto &[p, arg] :
			llvm::zip(branch.getDestOperands(), jmp_block->getArguments())) {
			block_parameters_to_args.emplace_back(p, arg);
		}
	}

	void handle_for_iter(mlir::emitpybytecode::ForIter for_iter,
		BlockInfo &info,
		std::vector<std::pair<std::variant<mlir::Value, ForwardedOutput>, mlir::BlockArgument>>
			&block_parameters_to_args,
		std::shared_ptr<spdlog::logger> &logger)
	{
		// ForIter forwards a loop variable to the body block
		ForwardedOutput loop_var{ for_iter, 0 };
		mlir::BlockArgument body_arg = for_iter.getBody()->getArgument(0);

		block_parameters_to_args.emplace_back(loop_var, body_arg);
		info.forwarded_outputs.push_back(loop_var);

		// The iterator is a use (it's an operand of FOR_ITER)
		// This is already captured by the operand processing above

		logger->debug("ForIter creates loop variable forwarded to body block argument");
	}

	/**
	 * Build alive_at_timestep from mlir::Liveness's block live-out info.
	 *
	 * Computes precise per-operation liveness by propagating backward within each block.
	 * This ensures values are only marked alive when actually needed, not conservatively
	 * throughout the entire block.
	 */
	void build_timesteps(const std::vector<mlir::Block *> &sorted_blocks,
		const std::map<mlir::Block *, BlockInfo> &block_info,
		const mlir::Liveness &liveness,
		std::map<mlir::Block *, std::pair<size_t, size_t>> &blocks_span)
	{
		auto logger = get_regalloc_logger();

		for (auto *block : sorted_blocks) {
			const auto &info = block_info.at(block);
			const auto start = alive_at_timestep.size();

			// Compute precise liveness by propagating backward within this block
			// alive_before[i] = (alive_after[i] - def[i]) ∪ use[i]
			std::vector<ValueSet> alive_before_op;
			alive_before_op.resize(info.operations.size());

			// Start from LiveOut (values alive at block exit) and work backward.
			// mlir::Liveness returns a SmallPtrSet<Value>; the project's
			// ValueSet is a variant<Value, ForwardedOutput> set, so we
			// convert. ForwardedOutputs aren't part of mlir::Liveness's view
			// and are added separately below.
			ValueSet alive_after;
			for (mlir::Value v : liveness.getLiveOut(block)) { alive_after.insert(v); }

			for (size_t i = info.operations.size(); i-- > 0;) {
				auto *op = info.operations[i];

				// Start with what's alive after this operation
				alive_before_op[i] = alive_after;

				// Remove values defined by this operation (they die here going backward)
				for (auto result : op->getResults()) { alive_before_op[i].erase(result); }

				// Add values used by this operation (they need to be alive going backward)
				for (auto operand : op->getOperands()) { alive_before_op[i].insert(operand); }

				// Propagate backward for next iteration
				alive_after = alive_before_op[i];
			}

			// For operations with side effects or that produce values in specific registers,
			// ensure their results are kept alive so they get proper register assignments.
			// This is needed for:
			// - Operations that raise exceptions (LoadAttribute, etc.)
			// - Operations that clobber r0 (CALL, YIELD, etc.) - their results MUST be tracked
			for (size_t i = 0; i < info.operations.size(); i++) {
				auto *op = info.operations[i];

				bool needs_tracking = !mlir::isPure(op);

				// CRITICAL: Always track CALL operations - their results go to r0 and MUST have
				// a live interval even if the result is unused
				if (!needs_tracking) {
					if (llvm::isa<mlir::emitpybytecode::FunctionCallOp>(op)
						|| llvm::isa<mlir::emitpybytecode::FunctionCallExOp>(op)
						|| llvm::isa<mlir::emitpybytecode::FunctionCallWithKeywordsOp>(op)) {
						needs_tracking = true;
					}
				}

				if (needs_tracking) {
					// Add results to the current operation's alive_before to ensure they appear
					// in the timestep where the operation executes. This is semantically odd
					// (the value doesn't exist before the operation), but it ensures the result
					// gets a register allocation at the point where it's produced.
					for (auto result : op->getResults()) { alive_before_op[i].insert(result); }
				}
			}

			// Add ForwardedOutputs to the first operation's alive set.
			// These are "defined" by the terminator but need to be live
			// throughout the block so the successor (the FOR_ITER body)
			// can pick the loop variable up via the same register. The
			// previous custom dataflow unconditionally injected these
			// into LiveIn before this point; mlir::Liveness doesn't know
			// about them, so they're injected here directly.
			if (!info.operations.empty()) {
				for (const auto &fwd : info.forwarded_outputs) { alive_before_op[0].insert(fwd); }
			}

			// Note: alive_before_op[0] may differ from mlir::Liveness's
			// LiveIn(block) because the needs_tracking pass above adds
			// impure operation results to the timestep of their defining
			// op. These results are defined within this block so they
			// cannot be in LiveIn. The discrepancy is intentional — it
			// ensures impure ops get a register assignment at their
			// definition site.

			// Now build timesteps in forward order using the computed liveness
			for (size_t i = 0; i < info.operations.size(); i++) {
				auto &alive = alive_at_timestep.emplace_back();

				// Add all values that are alive before this operation
				for (const auto &val : alive_before_op[i]) {
					std::visit([&alive](const auto &v) { alive.push_back(v); }, val);
				}
			}

			const auto end = alive_at_timestep.size();
			blocks_span.emplace(block, std::pair{ start, end });

			// Debug logging
			logger->debug("Block {} timesteps [{}, {}), {} ops, LiveIn has {} values",
				static_cast<void *>(block),
				start,
				end,
				info.operations.size(),
				liveness.getLiveIn(block).size());
			if (block->getTerminator()
				&& mlir::isa<mlir::emitpybytecode::ForIter>(block->getTerminator())) {
				logger->debug("  ^ FOR_ITER block");
			}
		}
	}

	/**
	 * Propagate block argument values through liveness information.
	 *
	 * This function transforms block arguments in the alive_at_timestep data into
	 * BlockArgumentInputs structures that track all the source values flowing into
	 * each block argument (PHI nodes in SSA form).
	 *
	 * For example, if:
	 *   bb1: br ^bb3(%val1)
	 *   bb2: br ^bb3(%val2)
	 *   bb3(%arg):
	 *
	 * Then %arg will be transformed into BlockArgumentInputs{%arg, [%val1, %val2]}
	 * indicating that %arg can receive values from either %val1 or %val2 depending
	 * on which predecessor block was executed.
	 *
	 * This information is crucial for register allocation to ensure that:
	 * 1. All source values are allocated to compatible registers
	 * 2. The block argument is allocated to the same register as its sources
	 * 3. MOVE instructions are inserted if sources end up in different registers
	 */
	void propagate_block_arguments(
		const std::vector<std::pair<std::variant<mlir::Value, ForwardedOutput>,
			mlir::BlockArgument>> &block_parameters_to_args,
		const std::map<mlir::Block *, std::pair<size_t, size_t>> &blocks_span)
	{
		// For each (source_value, block_argument) pair collected during block analysis
		for (const auto &[param, arg] : block_parameters_to_args) {
			auto *bb = arg.getOwner();
			const auto [start, end] = blocks_span.at(bb);
			auto block_timesteps =
				std::span{ alive_at_timestep.begin() + start, alive_at_timestep.begin() + end };

			// Replace all occurrences of the block argument with BlockArgumentInputs
			for (auto &ts : block_timesteps) {
				for (auto &val : ts) {
					// Check if this is the block argument we're looking for
					if (std::holds_alternative<mlir::Value>(val)
						&& mlir::isa<mlir::BlockArgument>(std::get<mlir::Value>(val))
						&& mlir::cast<mlir::BlockArgument>(std::get<mlir::Value>(val)) == arg) {
						// First occurrence: create BlockArgumentInputs with this source
						val = BlockArgumentInputs{ arg, { param } };
						block_input_mappings[param].insert(arg);
					} else if (std::holds_alternative<BlockArgumentInputs>(val)
							   && std::get<0>(std::get<BlockArgumentInputs>(val)) == arg) {
						// Subsequent occurrence: append this source to existing BlockArgumentInputs
						std::get<1>(std::get<BlockArgumentInputs>(val)).push_back(param);
						block_input_mappings[param].insert(arg);
					}
				}
			}
		}
	}

	/**
	 * Resolve transitive chains of block arguments.
	 *
	 * This function handles cases where a block argument receives values from other
	 * block arguments (transitive PHI nodes). For example:
	 *
	 *   bb1: br ^bb2(%val1)
	 *   bb2(%arg2): br ^bb3(%arg2)
	 *   bb3(%arg3):
	 *
	 * Here %arg3 receives %arg2, and %arg2 receives %val1. We need to resolve this
	 * chain so that %arg3 is understood to ultimately receive %val1.
	 *
	 * The function works backwards through timesteps, following chains of block
	 * arguments until reaching concrete values, and updates the block_input_mappings
	 * accordingly.
	 */
	void resolve_block_argument_chains()
	{
		// Process timesteps in reverse order
		for (auto &values : alive_at_timestep | std::views::reverse) {
			for (auto &value : values | std::views::reverse) {
				// Convert BlockArgumentInputs back to plain block arguments for processing
				if (std::holds_alternative<BlockArgumentInputs>(value)) {
					value = std::get<0>(std::get<BlockArgumentInputs>(value));
				}

				// Find if this value maps to any block arguments
				auto start =
					std::visit(overloaded{
								   [this](const auto &v) { return block_input_mappings.find(v); },
								   [this](const BlockArgumentInputs &) {
									   // BlockArgumentInputs are converted to plain mlir::Value
									   // (block arguments) at lines 485-487 before this visitor
									   // runs, so this branch is unreachable.
									   ASSERT(false);
									   return block_input_mappings.end();
								   },
							   },
						value);

				// Follow the chain of block arguments
				auto it = start;
				while (it != block_input_mappings.end()) {
					ASSERT(it->second.size() == 1);
					value = *it->second.begin();
					start->second.erase(start->second.begin());
					start->second.insert(
						mlir::cast<mlir::BlockArgument>(std::get<mlir::Value>(value)));
					it = block_input_mappings.find(std::get<mlir::Value>(value));
				}
			}
		}
	}
};

}// namespace codegen
