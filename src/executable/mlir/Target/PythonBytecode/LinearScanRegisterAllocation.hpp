#pragma once

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "LiveIntervalAnalysis.hpp"
#include "RegisterAllocationLogger.hpp"
#include "RegisterAllocationTypes.hpp"

#include "mlir/IR/Builders.h"
#include "utilities.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <bitset>
#include <iostream>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <variant>

namespace codegen {

/**
 * LinearScanRegisterAllocation implements a linear scan register allocation algorithm.
 *
 * The algorithm:
 * 1. Process live intervals in order of start position
 * 2. Expire old intervals and free their registers
 * 3. Allocate a free register to the current interval
 * 4. Handle special cases (r0 clobbering, ForIter constraints, GetIter constraints)
 */
class LinearScanRegisterAllocation
{
  public:
	// Register location
	struct Reg
	{
		size_t idx;
	};

	// Stack spill location (reserved for future use; iterative spilling via STORE_FAST/LOAD_FAST
	// ensures all final assignments are Reg, so StackLocation never appears in value2mem_map
	// after a completed allocation pass).
	struct StackLocation
	{
		size_t idx;
	};

	using ValueLocation = std::variant<Reg, StackLocation>;

	// Maps values to their allocated locations
	ValueMapping<ValueLocation> value2mem_map;

	using LiveIntervalSet = std::multiset<LiveIntervalAnalysis::LiveInterval,
		decltype([](const auto &lhs, const auto &rhs) { return lhs.end() < rhs.end(); })>;

	static constexpr size_t kRegCount = kNumRegisters;

	// Live interval analysis results (stored for visualization)
	std::optional<LiveIntervalAnalysis> live_interval_analysis;

	// Monotonically increasing across passes to generate unique spill slot names
	size_t m_spill_slot_count{ 0 };

	// Set to true by spill_value(); causes analyse() to restart the pass
	bool m_spills_emitted{ false };

	/**
	 * Run register allocation on the function.
	 *
	 * Uses an iterative strategy: if register pressure forces a spill, spill code
	 * (STORE_FAST / LOAD_FAST) is inserted into the IR and the entire analysis is
	 * restarted so that the new ops receive proper live intervals and register assignments.
	 * This repeats until a complete allocation with no spills is achieved.
	 */
	void analyse(mlir::func::FuncOp &func, mlir::OpBuilder builder)
	{
		m_spill_slot_count = 0;
		do {
			m_spills_emitted = false;
			value2mem_map.clear();
			live_interval_analysis.reset();
			run_single_pass(func, builder);
		} while (m_spills_emitted);
	}

  private:
	/**
	 * Single pass of liveness analysis + linear scan. Invoked by analyse().
	 * Returns without completing allocation if a spill is needed (m_spills_emitted is set).
	 */
	void run_single_pass(mlir::func::FuncOp &func, mlir::OpBuilder &builder)
	{
		auto logger = get_regalloc_logger();
		logger->info("Starting linear scan register allocation pass");

		// Run live interval analysis
		live_interval_analysis = LiveIntervalAnalysis{};
		live_interval_analysis->analyse(func);

		// Prepare for linear scan
		auto unhandled = std::span(live_interval_analysis->sorted_live_intervals.begin(),
			live_interval_analysis->sorted_live_intervals.end());

		ASSERT(std::is_sorted(unhandled.begin(),
			unhandled.end(),
			[](const LiveIntervalAnalysis::LiveInterval &lhs,
				const LiveIntervalAnalysis::LiveInterval &rhs) {
				return lhs.start() < rhs.start();
			}));

		LiveIntervalSet active;
		LiveIntervalSet inactive;
		LiveIntervalSet handled;

		// kNumRegisters available registers
		std::bitset<kRegCount> free;
		free.set();

		// Pre-allocate r0 for operations that clobber it
		preallocate_r0_clobbering_operations(unhandled, *live_interval_analysis, inactive);

		// Main linear scan loop
		while (!unhandled.empty()) {
			const auto &cur = *unhandled.begin();
			unhandled = unhandled.subspan(1, unhandled.size() - 1);

			logger->trace(
				"Processing interval: {} [start={}, end={}]", cur.value, cur.start(), cur.end());

			// Log active intervals before expiring
			if (logger->should_log(spdlog::level::trace)) {
				logger->trace("Active intervals before expire:");
				for (const auto &interval : active) {
					if (auto it = value2mem_map.find(interval.value); it != value2mem_map.end()) {
						if (std::holds_alternative<Reg>(it->second)) {
							logger->trace("  {} in r{} [start={}, end={}]",
								interval.value,
								std::get<Reg>(it->second).idx,
								interval.start(),
								interval.end());
						}
					}
				}
			}

			// Expire old intervals and free their registers
			bool was_inactive = expire_old_intervals(cur, active, inactive, handled, free, logger);

			// Collect available registers for this interval
			auto available_regs = collect_available_registers(
				cur, free, inactive, unhandled, *live_interval_analysis);

			if (available_regs.none()) {
				logger->info("Register pressure: spilling for {}", cur.value);
				spill_value(cur, active, func, builder);
				return;// Restart from scratch with updated IR
			} else {
				allocate_register(cur,
					free,
					available_regs,
					builder,
					*live_interval_analysis,
					active,
					was_inactive,
					logger);
			}
		}

		// Propagate register assignments to block arguments
		propagate_register_assignments(*live_interval_analysis);

		logger->info("Register allocation complete");
		log_register_assignments(logger);

		// Print visualization tables if requested via environment variable
		if (std::getenv("REGALLOC_VISUALIZE")) {
			print_liveness_table();
			print_register_allocation_table();
		}
	}

	/**
	 * Returns true if the interval's value is a GET_ITER result.
	 * GET_ITER intervals are deliberately collapsed to contiguous spans and must never be spilled.
	 */
	static bool is_get_iter_result(const LiveIntervalAnalysis::LiveInterval &interval)
	{
		if (!std::holds_alternative<mlir::Value>(interval.value)) { return false; }
		auto val = std::get<mlir::Value>(interval.value);
		return val.getDefiningOp() && mlir::isa<mlir::emitpybytecode::GetIter>(val.getDefiningOp());
	}

	/**
	 * Returns true if the interval's value is a regular (non-block-argument) op result that
	 * can be spilled using the standard STORE_FAST-after-def / LOAD_FAST-before-use pattern.
	 */
	static bool is_regular_op_result(const LiveIntervalAnalysis::LiveInterval &interval)
	{
		if (!std::holds_alternative<mlir::Value>(interval.value)) { return false; }
		auto val = std::get<mlir::Value>(interval.value);
		return !mlir::isa<mlir::BlockArgument>(val);
	}

	/**
	 * Spill a regular op-result value: insert STORE_FAST after its defining op and
	 * LOAD_FAST before each existing use, replacing those uses with the reload.
	 */
	void do_spill_op_result(mlir::Value victim_value,
		mlir::StringAttr name_attr,
		mlir::OpBuilder &builder)
	{
		// Collect existing uses before inserting the STORE_FAST (which adds a new use)
		llvm::SmallVector<mlir::OpOperand *> uses;
		for (auto &use : victim_value.getUses()) { uses.push_back(&use); }

		// After the defining op: STORE_FAST to save the spilled value
		auto *def_op = victim_value.getDefiningOp();
		ASSERT(def_op);
		builder.setInsertionPointAfter(def_op);
		builder.create<mlir::emitpybytecode::StoreFastOp>(
			def_op->getLoc(), name_attr, victim_value);

		// Before each original use: LOAD_FAST to reload the spilled value
		for (auto *use : uses) {
			auto *user_op = use->getOwner();
			builder.setInsertionPoint(user_op);
			auto reload = builder.create<mlir::emitpybytecode::LoadFastOp>(
				user_op->getLoc(), victim_value.getType(), name_attr);
			use->set(reload.getOutput());
		}
	}

	/**
	 * Spill a block argument: insert STORE_FAST as the first op of its block and
	 * LOAD_FAST before each use, replacing those uses with the reload.
	 *
	 * Used for both regular block arguments and ForwardedOutput loop variables
	 * (whose corresponding value is the body block's first argument).
	 */
	void do_spill_block_argument(mlir::BlockArgument arg,
		mlir::StringAttr name_attr,
		mlir::OpBuilder &builder)
	{
		auto *bb = arg.getOwner();
		ASSERT(!bb->empty());
		auto loc = bb->front().getLoc();

		// Collect existing uses before inserting STORE_FAST
		llvm::SmallVector<mlir::OpOperand *> uses;
		for (auto &use : arg.getUses()) { uses.push_back(&use); }

		// Insert STORE_FAST at the very start of the block
		builder.setInsertionPoint(bb, bb->begin());
		builder.create<mlir::emitpybytecode::StoreFastOp>(loc, name_attr, arg);

		// Before each original use: LOAD_FAST to reload the value
		for (auto *use : uses) {
			auto *user_op = use->getOwner();
			builder.setInsertionPoint(user_op);
			auto reload = builder.create<mlir::emitpybytecode::LoadFastOp>(
				user_op->getLoc(), arg.getType(), name_attr);
			use->set(reload.getOutput());
		}
	}

	/**
	 * Spill a value to a named local variable slot to free up a register.
	 *
	 * Victim selection (in priority order):
	 *   1. The active non-block-arg op-result with the latest end point (cheapest to spill).
	 *   2. Active block argument or ForwardedOutput (spilled at block entry).
	 *   3. cur itself if the above don't apply.
	 *
	 * GET_ITER results are never eligible: their intervals are collapsed to contiguous spans
	 * and must stay alive throughout the loop.
	 *
	 * After inserting spill code, sets m_spills_emitted = true so analyse() restarts.
	 */
	void spill_value(const LiveIntervalAnalysis::LiveInterval &cur,
		LiveIntervalSet &active,
		mlir::func::FuncOp &func,
		mlir::OpBuilder &builder)
	{
		auto logger = get_regalloc_logger();

		// --- Victim selection ---
		// Pass 1: prefer regular op-results (cheapest path)
		const LiveIntervalAnalysis::LiveInterval *victim_interval = nullptr;
		for (auto it = active.rbegin(); it != active.rend(); ++it) {
			if (!is_regular_op_result(*it)) { continue; }
			if (is_get_iter_result(*it)) { continue; }
			victim_interval = &(*it);
			break;
		}

		// Pass 2: fall back to block arguments / ForwardedOutputs
		if (!victim_interval) {
			for (auto it = active.rbegin(); it != active.rend(); ++it) {
				if (is_get_iter_result(*it)) { continue; }
				victim_interval = &(*it);
				break;
			}
		}

		// --- Choose: spill active victim or cur ---
		const LiveIntervalAnalysis::LiveInterval *to_spill = nullptr;
		if (victim_interval && victim_interval->end() > cur.end()) {
			to_spill = victim_interval;
			logger->info("Spilling active {} (end={}) to free register for {} (end={})",
				victim_interval->value,
				victim_interval->end(),
				cur.value,
				cur.end());
		} else if (!is_get_iter_result(cur)) {
			to_spill = &cur;
			logger->info("Spilling current {} (end={})", cur.value, cur.end());
		} else {
			logger->error(
				"Cannot spill: all live intervals are GET_ITER results — function requires "
				"more than {} registers.",
				kRegCount);
			TODO();
			return;
		}

		// --- Allocate spill slot ---
		const std::string spill_name = "__spill_" + std::to_string(m_spill_slot_count++);
		logger->info("Spilling {} to slot '{}'", to_spill->value, spill_name);

		auto *ctx = func->getContext();
		auto name_attr = mlir::StringAttr::get(ctx, spill_name);
		{
			auto existing = func->getAttr("locals");
			llvm::SmallVector<mlir::Attribute> locals;
			if (existing) {
				auto arr = mlir::cast<mlir::ArrayAttr>(existing);
				locals.assign(arr.begin(), arr.end());
			}
			locals.push_back(name_attr);
			func->setAttr("locals", mlir::ArrayAttr::get(ctx, locals));
		}

		// --- Dispatch spill by value type ---
		if (std::holds_alternative<mlir::Value>(to_spill->value)) {
			auto val = std::get<mlir::Value>(to_spill->value);
			if (mlir::isa<mlir::BlockArgument>(val)) {
				// Block argument: spill at block entry
				do_spill_block_argument(mlir::cast<mlir::BlockArgument>(val), name_attr, builder);
			} else {
				// Regular op result: spill after defining op
				do_spill_op_result(val, name_attr, builder);
			}
		} else {
			// ForwardedOutput: the loop variable from FOR_ITER lives as the body block's arg
			ASSERT(std::holds_alternative<ForwardedOutput>(to_spill->value));
			auto [op_ptr, idx] = std::get<ForwardedOutput>(to_spill->value);
			auto for_iter = mlir::cast<mlir::emitpybytecode::ForIter>(op_ptr);
			auto body_arg = for_iter.getBody()->getArgument(idx);
			do_spill_block_argument(body_arg, name_attr, builder);
		}

		m_spills_emitted = true;
	}

	/**
	 * Pre-allocate r0 for operations that directly clobber it (CALL, YIELD, etc.)
	 *
	 * This ensures that values produced by these operations are assigned to r0,
	 * matching the VM's behavior where these operations place results directly in r0.
	 *
	 * Note: Block arguments are NOT pre-allocated here. If a block argument receives
	 * values from r0-clobbering operations in different registers, MOVE instructions
	 * will be inserted at block boundaries during bytecode emission.
	 */
	void preallocate_r0_clobbering_operations(
		std::span<LiveIntervalAnalysis::LiveInterval> unhandled,
		const LiveIntervalAnalysis &live_interval_analysis,
		LiveIntervalSet &inactive)
	{
		auto logger = get_regalloc_logger();

		for (const auto &interval : unhandled) {
			// Only pre-allocate values that directly clobber r0
			if (!std::holds_alternative<mlir::Value>(interval.value)) { continue; }

			auto value = std::get<mlir::Value>(interval.value);

			// Skip block arguments - they don't clobber r0 themselves
			if (mlir::isa<mlir::BlockArgument>(value)) { continue; }

			if (clobbers_r0(value)) {
				// Pre-allocate r0 but DON'T add to inactive - let it be processed normally
				// in the main loop to handle conflicts and spilling if needed
				value2mem_map.insert_or_assign(interval.value, Reg{ .idx = 0 });
				logger->debug("Pre-allocated r0 for: {}", interval.value);
			}
		}
	}

	/**
	 * Expire intervals that are no longer alive and free their registers
	 * Returns true if cur was found in inactive and moved to active
	 */
	bool expire_old_intervals(const LiveIntervalAnalysis::LiveInterval &cur,
		LiveIntervalSet &active,
		LiveIntervalSet &inactive,
		LiveIntervalSet &handled,
		std::bitset<kRegCount> &free,
		std::shared_ptr<spdlog::logger> &logger)
	{
		bool cur_was_inactive = false;

		// Expire active intervals
		for (auto it = active.begin(); it != active.end();) {
			const auto &interval = *it;
			ASSERT(interval.value != cur.value);

			if (interval.end() < cur.start()) {
				// Interval completely expired
				handled.insert(interval);
				it = active.erase(it);
				free_register(interval, free, logger);
			} else if (!interval.alive_at(cur.start())) {
				// Interval temporarily not alive (goes inactive).
				// GET_ITER intervals are guaranteed contiguous by extend_iterator_liveness(),
				// so they will never take this branch during their loop span.
				inactive.insert(interval);
				it = active.erase(it);
				free_register(interval, free, logger);
			} else {
				++it;
			}
		}

		// Reactivate or expire inactive intervals
		for (auto it = inactive.begin(); it != inactive.end();) {
			const auto &interval = *it;

			if (interval.value == cur.value) {
				// Current interval was previously allocated (e.g., r0 clobbering)
				active.insert(interval);
				it = inactive.erase(it);
				cur_was_inactive = true;
				logger->debug("Moved cur from inactive to active: {}", cur.value);
			} else if (interval.end() < cur.start()) {
				// Interval completely expired
				handled.insert(interval);
				it = inactive.erase(it);
			} else if (interval.alive_at(cur.start())) {
				// Interval becomes active again
				active.insert(interval);
				it = inactive.erase(it);
				mark_register_used(interval, free, logger);
				logger->debug("Reactivated interval: {}", interval.value);
			} else {
				++it;
			}
		}

		return cur_was_inactive;
	}

	/**
	 * Collect available registers, accounting for special constraints
	 */
	std::bitset<kRegCount> collect_available_registers(
		const LiveIntervalAnalysis::LiveInterval &cur,
		const std::bitset<kRegCount> &free,
		LiveIntervalSet &inactive,
		std::span<LiveIntervalAnalysis::LiveInterval> unhandled,
		const LiveIntervalAnalysis &live_interval_analysis)
	{
		auto logger = get_regalloc_logger();
		auto available = free;

		// Exclude registers used by overlapping inactive intervals
		auto overlaps =
			std::views::filter([&cur](const auto &interval) { return interval.overlaps(cur); });

		for (const auto &interval : inactive | overlaps) {
			if (auto it = value2mem_map.find(interval.value); it != value2mem_map.end()) {
				const auto reg = it->second;
				ASSERT(std::holds_alternative<Reg>(reg));
				available.set(std::get<Reg>(reg).idx, false);
			}
		}

		// Exclude registers used by overlapping unhandled intervals
		for (const auto &interval : unhandled | overlaps) {
			if (auto it = value2mem_map.find(interval.value); it != value2mem_map.end()) {
				const auto reg = it->second;
				ASSERT(std::holds_alternative<Reg>(reg));
				available.set(std::get<Reg>(reg).idx, false);
			}
		}

		// Apply special constraints
		apply_special_constraints(cur, available, live_interval_analysis);

		return available;
	}

	/**
	 * Apply special constraints for specific operations:
	 * - GetIter cannot use r0 (reserved for function call results by VM convention)
	 * - BuildList cannot use r0 (ListExtend internally calls the iterator protocol
	 *   which executes Python bytecode and triggers pop_frame(true), propagating
	 *   the callee's r0 into the caller's r0, overwriting the list)
	 */
	void apply_special_constraints(const LiveIntervalAnalysis::LiveInterval &cur,
		std::bitset<kRegCount> &available,
		const LiveIntervalAnalysis & /*live_interval_analysis*/)
	{
		auto logger = get_regalloc_logger();

		if (std::holds_alternative<mlir::Value>(cur.value)) {
			auto value = std::get<mlir::Value>(cur.value);
			// GetIter: cannot use r0 (r0 is reserved for function call results)
			if (value.getDefiningOp()
				&& mlir::isa<mlir::emitpybytecode::GetIter>(value.getDefiningOp())) {
				available.set(0, false);
				logger->debug("GetIter result cannot use r0");
			}
			// BuildList: cannot use r0 (ListExtend iterates Python iterators which
			// trigger pop_frame(true) and overwrite r0 with the iterator's return value)
			if (value.getDefiningOp()
				&& mlir::isa<mlir::emitpybytecode::BuildList>(value.getDefiningOp())) {
				available.set(0, false);
				logger->debug("BuildList result cannot use r0");
			}
		}
	}

	/**
	 * Allocate a register for the current interval
	 */
	void allocate_register(const LiveIntervalAnalysis::LiveInterval &cur,
		std::bitset<kRegCount> &free,
		const std::bitset<kRegCount> &available,
		mlir::OpBuilder &builder,
		const LiveIntervalAnalysis &live_interval_analysis,
		LiveIntervalSet &active,
		bool was_inactive,
		std::shared_ptr<spdlog::logger> &logger)
	{
		std::optional<size_t> cur_reg;

		// Check if already allocated (from pre-allocation phase)
		if (auto it = value2mem_map.find(cur.value); it != value2mem_map.end()) {
			ASSERT(std::holds_alternative<Reg>(it->second));
			cur_reg = std::get<Reg>(it->second).idx;
			logger->trace("Using pre-allocated register r{}", *cur_reg);
		} else {
			// Find first available register
			for (size_t i = 0; i < available.size(); ++i) {
				if (available.test(i)) {
					cur_reg = i;
					value2mem_map.insert_or_assign(cur.value, Reg{ .idx = i });
					logger->debug("Allocated r{} to {}", i, cur.value);
					break;
				}
			}
		}

		ASSERT(cur_reg.has_value());

		// Handle case where the chosen register is not free (need to save/restore)
		if (!free.test(*cur_reg)) {
			logger->info("Register conflict: r{} is not free, handling conflict for {}",
				*cur_reg,
				cur.value);
			handle_register_conflict(
				cur, *cur_reg, available, free, builder, live_interval_analysis, logger);
		} else {
			logger->debug("Marking r{} as not free for {} [{}..{})",
				*cur_reg,
				cur.value,
				cur.start(),
				cur.end());
			free.set(*cur_reg, false);
		}

		// Only insert into active if it wasn't already moved from inactive
		if (!was_inactive) {
			active.insert(cur);
			logger->debug(
				"Added to active: {} in r{} [{}..{})", cur.value, *cur_reg, cur.start(), cur.end());
		} else {
			logger->debug("Already in active (was inactive): {} in r{} [{}..{})",
				cur.value,
				*cur_reg,
				cur.start(),
				cur.end());
		}
	}

	/**
	 * Handle the case where we need a register that's currently in use
	 * by inserting save/restore code
	 */
	void handle_register_conflict(const LiveIntervalAnalysis::LiveInterval &cur,
		size_t cur_reg,
		const std::bitset<kRegCount> &available,
		std::bitset<kRegCount> &free,
		mlir::OpBuilder &builder,
		const LiveIntervalAnalysis &live_interval_analysis,
		std::shared_ptr<spdlog::logger> &logger)
	{
		// Find a scratch register
		std::optional<size_t> scratch_reg;
		for (size_t i = 1; i < available.size(); ++i) {
			if (available.test(i)) {
				scratch_reg = i;
				break;
			}
		}
		ASSERT(scratch_reg.has_value());

		if (std::holds_alternative<mlir::Value>(cur.value)) {
			auto current_value = std::get<mlir::Value>(cur.value);

			// Resolve block arguments to their defining operations
			if (mlir::isa<mlir::BlockArgument>(current_value)) {
				if (auto it = live_interval_analysis.block_input_mappings.find(cur.value);
					it != live_interval_analysis.block_input_mappings.end()) {
					for (auto mapped_value : it->second) {
						ASSERT(!std::holds_alternative<ForwardedOutput>(mapped_value));
						if (clobbers_r0(std::get<mlir::Value>(mapped_value))) {
							ASSERT(mlir::isa<mlir::BlockArgument>(current_value));
							current_value = std::get<mlir::Value>(mapped_value);
							break;
						}
					}
				}
			}

			ASSERT(!mlir::isa<mlir::BlockArgument>(current_value));
			auto loc = current_value.getLoc();

			// Insert: push r{cur_reg}, move r{scratch}, r{cur_reg}, pop r{cur_reg}
			builder.setInsertionPoint(current_value.getDefiningOp());
			builder.create<mlir::emitpybytecode::Push>(loc, cur_reg);
			builder.setInsertionPointAfter(current_value.getDefiningOp());
			builder.create<mlir::emitpybytecode::Move>(loc, *scratch_reg, cur_reg);
			builder.create<mlir::emitpybytecode::Pop>(loc, cur_reg);

			value2mem_map.insert_or_assign(cur.value, Reg{ .idx = *scratch_reg });

			// BUG FIX: Mark the scratch register as not free
			free.set(*scratch_reg, false);

			logger->info(
				"Register conflict: moved {} from r{} to r{} (scratch), marked as not free",
				current_value,
				cur_reg,
				*scratch_reg);
		} else {
			// ForwardedOutput: the defining op is the FOR_ITER terminator.
			// The loop variable becomes available in the body block; insert PUSH before
			// FOR_ITER and MOVE/POP at the start of the body block.
			ASSERT(std::holds_alternative<ForwardedOutput>(cur.value));
			auto [op_ptr, idx] = std::get<ForwardedOutput>(cur.value);
			auto *for_iter_op = op_ptr;
			auto loc = for_iter_op->getLoc();

			auto for_iter = mlir::cast<mlir::emitpybytecode::ForIter>(for_iter_op);
			auto *body_block = for_iter.getBody();
			ASSERT(!body_block->empty());

			// Save cur_reg before FOR_ITER, move loop variable to scratch, restore cur_reg
			builder.setInsertionPoint(for_iter_op);
			builder.create<mlir::emitpybytecode::Push>(loc, cur_reg);
			builder.setInsertionPoint(body_block, body_block->begin());
			builder.create<mlir::emitpybytecode::Move>(loc, *scratch_reg, cur_reg);
			builder.create<mlir::emitpybytecode::Pop>(loc, cur_reg);

			value2mem_map.insert_or_assign(cur.value, Reg{ .idx = *scratch_reg });
			free.set(*scratch_reg, false);

			logger->info(
				"Register conflict (ForwardedOutput): moved FOR_ITER loop var from r{} to r{} "
				"(scratch)",
				cur_reg,
				*scratch_reg);
		}
	}

	/**
	 * Free a register when an interval expires
	 */
	void free_register(const LiveIntervalAnalysis::LiveInterval &interval,
		std::bitset<kRegCount> &free,
		std::shared_ptr<spdlog::logger> &logger)
	{
		const auto reg = value2mem_map.at(interval.value);
		ASSERT(std::holds_alternative<Reg>(reg));
		size_t reg_idx = std::get<Reg>(reg).idx;
		free.set(reg_idx, true);
		logger->debug("Freed r{} from {} [{}..{})",
			reg_idx,
			interval.value,
			interval.start(),
			interval.end());
	}

	/**
	 * Mark a register as used when an interval becomes active
	 */
	void mark_register_used(const LiveIntervalAnalysis::LiveInterval &interval,
		std::bitset<kRegCount> &free,
		std::shared_ptr<spdlog::logger> &logger)
	{
		const auto reg = value2mem_map.at(interval.value);
		ASSERT(std::holds_alternative<Reg>(reg));
		size_t reg_idx = std::get<Reg>(reg).idx;
		ASSERT(free.test(reg_idx));
		free.set(reg_idx, false);
		logger->trace("Marked r{} as used for {}", reg_idx, interval.value);
	}

	/**
	 * Propagate register assignments to block arguments
	 */
	void propagate_register_assignments(const LiveIntervalAnalysis &live_interval_analysis)
	{
		decltype(value2mem_map) additional_mappings;

		for (auto [value, reg] : value2mem_map) {
			if (auto it = live_interval_analysis.block_input_mappings.find(value);
				it != live_interval_analysis.block_input_mappings.end()) {
				for (auto mapped_value : it->second) { additional_mappings[mapped_value] = reg; }
			}
		}

		value2mem_map.merge(std::move(additional_mappings));
	}

	/**
	 * Log final register assignments
	 */
	void log_register_assignments(std::shared_ptr<spdlog::logger> &logger)
	{
		if (logger->should_log(spdlog::level::debug)) {
			logger->debug("Final register assignments:");
			for (const auto &[value, location] : value2mem_map) {
				if (std::holds_alternative<Reg>(location)) {
					logger->debug("  {} -> r{}", value, std::get<Reg>(location).idx);
				}
			}
		}
	}

	/**
	 * Print liveness visualization table
	 * Shows which values are alive at each timestep
	 */
	void print_liveness_table()
	{
		if (!live_interval_analysis.has_value()) {
			llvm::outs() << "Error: No live interval analysis available\n";
			return;
		}

		llvm::outs() << "\n=== LIVENESS VISUALIZATION ===\n\n";

		// Collect all values and sort them
		std::vector<std::variant<mlir::Value, ForwardedOutput>> all_values;
		for (const auto &interval : live_interval_analysis->sorted_live_intervals) {
			all_values.push_back(interval.value);
		}

		// Get max timesteps from intervals
		size_t max_timestep = 0;
		for (const auto &interval : live_interval_analysis->sorted_live_intervals) {
			max_timestep = std::max(max_timestep, interval.end());
		}

		// Print header
		llvm::outs() << "Value                                                   | ";
		for (size_t t = 0; t < max_timestep; ++t) { llvm::outs() << llvm::format("%3d", t); }
		llvm::outs() << "\n";

		// Print separator
		llvm::outs() << "--------------------------------------------------------+-";
		for (size_t t = 0; t < max_timestep; ++t) { llvm::outs() << "---"; }
		llvm::outs() << "\n";

		// Print each value's liveness
		for (const auto &interval : live_interval_analysis->sorted_live_intervals) {
			// Print value name (truncate to 55 chars)
			std::string value_str = fmt::format("{}", interval.value);
			if (value_str.length() > 55) { value_str = value_str.substr(0, 52) + "..."; }
			llvm::outs() << llvm::format("%-55s", value_str.c_str()) << " | ";

			// Print liveness for each timestep
			for (size_t t = 0; t < max_timestep; ++t) {
				if (interval.alive_at(t)) {
					llvm::outs() << "  x";
				} else {
					llvm::outs() << "   ";
				}
			}
			llvm::outs() << "\n";
		}

		llvm::outs() << "\n";
	}

	/**
	 * Print register allocation visualization table
	 * Shows which register each value is assigned to at each timestep
	 */
	void print_register_allocation_table()
	{
		if (!live_interval_analysis.has_value()) {
			llvm::outs() << "Error: No live interval analysis available\n";
			return;
		}

		llvm::outs() << "\n=== REGISTER ALLOCATION VISUALIZATION ===\n\n";

		// Get max timesteps from intervals
		size_t max_timestep = 0;
		for (const auto &interval : live_interval_analysis->sorted_live_intervals) {
			max_timestep = std::max(max_timestep, interval.end());
		}

		// Print header
		llvm::outs() << "Value                                                   | ";
		for (size_t t = 0; t < max_timestep; ++t) { llvm::outs() << llvm::format("%3d", t); }
		llvm::outs() << "\n";

		// Print separator
		llvm::outs() << "--------------------------------------------------------+-";
		for (size_t t = 0; t < max_timestep; ++t) { llvm::outs() << "---"; }
		llvm::outs() << "\n";

		// Print each value's register assignment
		for (const auto &interval : live_interval_analysis->sorted_live_intervals) {
			// Print value name (truncate to 55 chars)
			std::string value_str = fmt::format("{}", interval.value);
			if (value_str.length() > 55) { value_str = value_str.substr(0, 52) + "..."; }
			llvm::outs() << llvm::format("%-55s", value_str.c_str()) << " | ";

			// Find register for this value
			std::optional<size_t> reg_idx;
			if (auto it = value2mem_map.find(interval.value); it != value2mem_map.end()) {
				if (std::holds_alternative<Reg>(it->second)) {
					reg_idx = std::get<Reg>(it->second).idx;
				}
			}

			// Print register assignment for each timestep
			for (size_t t = 0; t < max_timestep; ++t) {
				if (interval.alive_at(t) && reg_idx.has_value()) {
					llvm::outs() << llvm::format(" r%d", *reg_idx);
				} else {
					llvm::outs() << "   ";
				}
			}
			llvm::outs() << "\n";
		}

		llvm::outs() << "\n";
	}
};

}// namespace codegen
