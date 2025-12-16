#pragma once

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "RegisterAllocationLogger.hpp"
#include "RegisterAllocationTypes.hpp"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
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
 * LiveAnalysis performs backward dataflow analysis to determine which values are alive
 * at each point in the program. This is the first step in register allocation.
 *
 * Uses the standard liveness algorithm:
 *   LiveOut[B] = union of LiveIn[S] for all successors S of B
 *   LiveIn[B] = Use[B] ∪ (LiveOut[B] - Def[B])
 *   Iterate until fixed point
 *
 * This correctly handles loops and back edges.
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
	 * Analyze the function to determine liveness information using backward dataflow
	 */
	void analyse(mlir::func::FuncOp &fn)
	{
		auto logger = get_regalloc_logger();
		logger->info(
			"Starting backward dataflow live analysis for function: {}", fn.getName().str());

		auto &region = fn.getRegion();
		auto sorted_blocks = sortBlocks(region);

		// Build block information and Use/Def sets
		std::map<mlir::Block *, BlockInfo> block_info;
		std::vector<std::pair<std::variant<mlir::Value, ForwardedOutput>, mlir::BlockArgument>>
			block_parameters_to_args;

		for (auto *block : sorted_blocks) {
			build_block_info(block, block_info[block], block_parameters_to_args, logger);
		}

		// Run backward dataflow to compute LiveIn/LiveOut
		compute_liveness(sorted_blocks, block_info, logger);

		// Build alive_at_timestep from LiveIn/LiveOut
		std::map<mlir::Block *, std::pair<size_t, size_t>> blocks_span;
		build_timesteps(sorted_blocks, block_info, blocks_span);

		// Propagate block argument inputs through the liveness information
		propagate_block_arguments(block_parameters_to_args, blocks_span);

		// Resolve transitive block argument mappings
		resolve_block_argument_chains();

		logger->info("Live analysis complete. Tracked {} timesteps", alive_at_timestep.size());
	}

  private:
	/**
	 * Information about a single block for dataflow analysis
	 */
	struct BlockInfo
	{
		// Values used in this block (before being defined)
		ValueSet use;

		// Values defined in this block
		ValueSet def;

		// Values live at entry to this block (computed by dataflow)
		ValueSet live_in;

		// Values live at exit from this block (computed by dataflow)
		ValueSet live_out;

		// Operations in this block (in order)
		std::vector<mlir::Operation *> operations;

		// ForwardedOutputs created by terminators in this block
		std::vector<ForwardedOutput> forwarded_outputs;
	};

	/**
	 * Build Use/Def sets for a block
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

		// Build Use/Def sets
		// For each operation, add operands to Use (if not already in Def), and add results to Def
		for (auto &op : block->getOperations()) {
			info.operations.push_back(&op);

			// Add operands to Use (if not already defined)
			for (const auto &operand : op.getOperands()) {
				if (!info.def.contains(operand)) { info.use.insert(operand); }
			}

			// Add results to Def
			for (const auto &result : op.getResults()) { info.def.insert(result); }
		}

		// Handle terminators specially
		if (auto *terminator = block->getTerminator()) {
			handle_terminator(terminator, info, block_parameters_to_args, logger);
		}

		logger->debug("Block has {} uses, {} defs", info.use.size(), info.def.size());
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
	 * Compute LiveIn/LiveOut using backward dataflow iteration
	 */
	void compute_liveness(const std::vector<mlir::Block *> &sorted_blocks,
		std::map<mlir::Block *, BlockInfo> &block_info,
		std::shared_ptr<spdlog::logger> &logger)
	{
		logger->info("Running backward dataflow iteration to compute liveness");

		// Initialize all LiveIn and LiveOut to empty (already done by default)

		// Iterate until fixed point
		bool changed = true;
		int iteration = 0;

		while (changed) {
			changed = false;
			iteration++;

			logger->debug("Dataflow iteration {}", iteration);

			// Process blocks in reverse post-order for better convergence
			for (auto it = sorted_blocks.rbegin(); it != sorted_blocks.rend(); ++it) {
				auto *block = *it;
				auto &info = block_info[block];

				// Save old LiveIn for convergence check
				auto old_live_in = info.live_in;

				// LiveOut[B] = union of LiveIn[S] for all successors S
				info.live_out.clear();
				for (auto *successor : block->getSuccessors()) {
					const auto &succ_info = block_info[successor];
					info.live_out.insert(succ_info.live_in.begin(), succ_info.live_in.end());
				}

				// LiveIn[B] = Use[B] ∪ (LiveOut[B] - Def[B])
				info.live_in = info.use;
				for (const auto &val : info.live_out) {
					if (!info.def.contains(val)) { info.live_in.insert(val); }
				}

				// Add ForwardedOutputs to LiveIn (they're "defined" by the terminator but need
				// to be live for the successor)
				for (const auto &fwd : info.forwarded_outputs) { info.live_in.insert(fwd); }

				// Check if LiveIn changed
				if (info.live_in != old_live_in) { changed = true; }
			}
		}

		logger->info("Dataflow converged after {} iterations", iteration);

		// Debug: print LiveIn/LiveOut for each block
		for (auto *block : sorted_blocks) {
			const auto &info = block_info[block];
			logger->debug("Block {} LiveIn: {} values, LiveOut: {} values",
				static_cast<void *>(block),
				info.live_in.size(),
				info.live_out.size());
		}
	}

	/**
	 * Build alive_at_timestep from LiveIn/LiveOut
	 *
	 * Computes precise per-operation liveness by propagating backward within each block.
	 * This ensures values are only marked alive when actually needed, not conservatively
	 * throughout the entire block.
	 */
	void build_timesteps(const std::vector<mlir::Block *> &sorted_blocks,
		const std::map<mlir::Block *, BlockInfo> &block_info,
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

			// Start from LiveOut (values alive at block exit) and work backward
			ValueSet alive_after = info.live_out;

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

			// For operations with side effects, ensure their results are kept alive
			// even if they're not used, so they get proper register assignments.
			// This is needed for operations like LoadAttribute that may raise exceptions
			// or trigger descriptors, and must be emitted even if results are unused.
			for (size_t i = 0; i < info.operations.size(); i++) {
				auto *op = info.operations[i];

				// Check if operation is pure (has no side effects)
				// Non-pure operations must be emitted even if results are unused
				if (!mlir::isPure(op)) {
					// Re-add any results to ensure they get register assignments
					for (auto result : op->getResults()) { alive_before_op[i].insert(result); }
				}
			}

			// Add ForwardedOutputs to the first operation if they're in LiveIn
			// (they're "defined" by the terminator but need to be live for the successor)
			if (!info.operations.empty()) {
				for (const auto &fwd : info.forwarded_outputs) {
					if (info.live_in.contains(fwd)) { alive_before_op[0].insert(fwd); }
				}
			}

			// Sanity check: alive_before[0] should equal LiveIn
			// (We computed it backward from LiveOut, should match forward computation)
			if (!alive_before_op.empty() && alive_before_op[0] != info.live_in) {
				logger->warn(
					"Block {} liveness mismatch: alive_before[0] has {} values, LiveIn has {} "
					"values",
					static_cast<void *>(block),
					alive_before_op[0].size(),
					info.live_in.size());

				// Debug: show the difference
				logger->debug("  Values in LiveIn but not alive_before[0]:");
				for (const auto &val : info.live_in) {
					if (!alive_before_op[0].contains(val)) {
						logger->debug("    {}", to_string(val));
					}
				}
				logger->debug("  Values in alive_before[0] but not LiveIn:");
				for (const auto &val : alive_before_op[0]) {
					if (!info.live_in.contains(val)) { logger->debug("    {}", to_string(val)); }
				}
			}

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
				info.live_in.size());
			if (block->getTerminator()
				&& mlir::isa<mlir::emitpybytecode::ForIter>(block->getTerminator())) {
				logger->debug("  ^ FOR_ITER block");
			}
		}
	}

	/**
	 * Propagate block argument values through liveness information
	 */
	void propagate_block_arguments(
		const std::vector<std::pair<std::variant<mlir::Value, ForwardedOutput>,
			mlir::BlockArgument>> &block_parameters_to_args,
		const std::map<mlir::Block *, std::pair<size_t, size_t>> &blocks_span)
	{
		for (const auto &[param, arg] : block_parameters_to_args) {
			auto *bb = arg.getOwner();
			const auto [start, end] = blocks_span.at(bb);
			auto block_timesteps =
				std::span{ alive_at_timestep.begin() + start, alive_at_timestep.begin() + end };

			for (auto &ts : block_timesteps) {
				for (auto &val : ts) {
					if (std::holds_alternative<mlir::Value>(val)
						&& mlir::isa<mlir::BlockArgument>(std::get<mlir::Value>(val))
						&& mlir::cast<mlir::BlockArgument>(std::get<mlir::Value>(val)) == arg) {
						val = BlockArgumentInputs{ arg, { param } };
						block_input_mappings[param].insert(arg);
					} else if (std::holds_alternative<BlockArgumentInputs>(val)
							   && std::get<0>(std::get<BlockArgumentInputs>(val)) == arg) {
						std::get<1>(std::get<BlockArgumentInputs>(val)).push_back(param);
						block_input_mappings[param].insert(arg);
					}
				}
			}
		}
	}

	/**
	 * Resolve transitive chains of block arguments
	 */
	void resolve_block_argument_chains()
	{
		for (auto &values : alive_at_timestep | std::views::reverse) {
			for (auto &value : values | std::views::reverse) {
				if (std::holds_alternative<BlockArgumentInputs>(value)) {
					value = std::get<0>(std::get<BlockArgumentInputs>(value));
				}

				auto start =
					std::visit(overloaded{
								   [this](const auto &v) { return block_input_mappings.find(v); },
								   [this](const BlockArgumentInputs &) {
									   TODO();
									   return block_input_mappings.end();
								   },
							   },
						value);

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
