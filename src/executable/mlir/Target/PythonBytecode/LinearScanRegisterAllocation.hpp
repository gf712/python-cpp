#pragma once

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "LiveIntervalAnalysis.hpp"
#include "RegisterAllocationLogger.hpp"
#include "RegisterAllocationTypes.hpp"

#include "mlir/IR/Builders.h"
#include "utilities.hpp"

#include "llvm/Support/Format.h"

#include <algorithm>
#include <bitset>
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

	// Stack spill location (currently not used - we abort if we run out of registers)
	struct StackLocation
	{
		size_t idx;
	};

	using ValueLocation = std::variant<Reg, StackLocation>;

	// Maps values to their allocated locations
	ValueMapping<ValueLocation> value2mem_map;

	using LiveIntervalSet = std::multiset<LiveIntervalAnalysis::LiveInterval,
		decltype([](const auto &lhs, const auto &rhs) { return lhs.end() < rhs.end(); })>;

	// Track registers that are reserved for FOR_ITER iterators
	// Maps loop variable value -> iterator register index
	ValueMapping<size_t> foriter_reserved_regs;

	// Live interval analysis results (stored for visualization)
	std::optional<LiveIntervalAnalysis> live_interval_analysis;

	/**
	 * Run register allocation on the function
	 */
	void analyse(mlir::func::FuncOp &func, mlir::OpBuilder builder)
	{
		auto logger = get_regalloc_logger();
		// Enable debug logging temporarily to diagnose ForIter bug
		logger->set_level(spdlog::level::debug);
		logger->info("Starting linear scan register allocation");

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

		// 32 available registers
		std::bitset<32> free;
		free.set();

		// Pre-allocate r0 for operations that clobber it
		preallocate_r0_clobbering_operations(unhandled, *live_interval_analysis, inactive);

		// Main linear scan loop
		while (!unhandled.empty()) {
			const auto &cur = *unhandled.begin();
			unhandled = unhandled.subspan(1, unhandled.size() - 1);

			logger->trace("Processing interval: {}", to_string(cur.value));

			// Expire old intervals and free their registers
			expire_old_intervals(cur, active, inactive, handled, free, logger);

			// Collect available registers for this interval
			auto available_regs =
				collect_available_registers(cur, free, inactive, unhandled, *live_interval_analysis);

			if (available_regs.none()) {
				logger->error("No available registers for {}", to_string(cur.value));
				TODO();// Should implement spilling
			} else {
				allocate_register(
					cur, free, available_regs, builder, *live_interval_analysis, active, logger);
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

  private:
	/**
	 * Pre-allocate register 0 for operations that must use it
	 * (function calls, yield, etc.)
	 */
	void preallocate_r0_clobbering_operations(
		std::span<LiveIntervalAnalysis::LiveInterval> unhandled,
		const LiveIntervalAnalysis &live_interval_analysis,
		LiveIntervalSet &inactive)
	{
		auto logger = get_regalloc_logger();

		for (const auto &interval : unhandled) {
			bool needs_r0 = false;

			// Check if this value directly clobbers r0
			if (std::holds_alternative<mlir::Value>(interval.value)) {
				auto value = std::get<mlir::Value>(interval.value);
				if (clobbers_r0(value)) {
					needs_r0 = true;
					logger->debug("Value {} clobbers r0", to_string(value));
				}
			}

			// Check if this value flows from something that clobbers r0
			if (!needs_r0) {
				if (auto it = live_interval_analysis.block_input_mappings.find(interval.value);
					it != live_interval_analysis.block_input_mappings.end()) {
					for (auto mapped_value : it->second) {
						if (std::holds_alternative<ForwardedOutput>(mapped_value)) { continue; }
						if (clobbers_r0(std::get<mlir::Value>(mapped_value))) {
							needs_r0 = true;
							logger->debug("Value {} flows from r0-clobbering value",
								to_string(interval.value));
							break;
						}
					}
				}
			}

			if (needs_r0) {
				value2mem_map.insert_or_assign(interval.value, Reg{ .idx = 0 });
				inactive.insert(interval);
			}
		}
	}

	/**
	 * Expire intervals that are no longer alive and free their registers
	 */
	void expire_old_intervals(const LiveIntervalAnalysis::LiveInterval &cur,
		LiveIntervalSet &active,
		LiveIntervalSet &inactive,
		LiveIntervalSet &handled,
		std::bitset<32> &free,
		std::shared_ptr<spdlog::logger> &logger)
	{
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
				// Interval temporarily not alive (goes inactive)
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
			} else if (interval.end() < cur.start()) {
				// Interval completely expired
				handled.insert(interval);
				it = inactive.erase(it);
			} else if (interval.alive_at(cur.start())) {
				// Interval becomes active again
				active.insert(interval);
				it = inactive.erase(it);
				mark_register_used(interval, free, logger);
			} else {
				++it;
			}
		}
	}

	/**
	 * Collect available registers, accounting for special constraints
	 */
	std::bitset<32> collect_available_registers(const LiveIntervalAnalysis::LiveInterval &cur,
		const std::bitset<32> &free,
		LiveIntervalSet &inactive,
		std::span<LiveIntervalAnalysis::LiveInterval> unhandled,
		const LiveIntervalAnalysis &live_interval_analysis)
	{
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
	 * - GetIter cannot use r0 (reserved for function call results)
	 * - ForIter loop variable cannot use the same register as its iterator
	 */
	void apply_special_constraints(const LiveIntervalAnalysis::LiveInterval &cur,
		std::bitset<32> &available,
		const LiveIntervalAnalysis &live_interval_analysis)
	{
		auto logger = get_regalloc_logger();

		// GetIter: cannot use r0
		if (std::holds_alternative<mlir::Value>(cur.value)) {
			auto value = std::get<mlir::Value>(cur.value);
			if (value.getDefiningOp()
				&& mlir::isa<mlir::emitpybytecode::GetIter>(value.getDefiningOp())) {
				available.set(0, false);
				logger->debug("GetIter result cannot use r0");
			}
		}

		// FIX FOR FORITER BUG:
		// ForIter loop variable (ForwardedOutput) cannot use same register as iterator
		if (std::holds_alternative<ForwardedOutput>(cur.value)) {
			auto forwarded = std::get<ForwardedOutput>(cur.value);
			if (auto for_iter = mlir::dyn_cast<mlir::emitpybytecode::ForIter>(forwarded.first)) {
				// Get the iterator value
				auto iterator = for_iter.getIterator();

				logger->debug("Processing ForIter loop variable, looking for iterator register");

				// Find what register the iterator is assigned to
				// Need to check both as mlir::Value and potentially through block argument mappings
				std::optional<size_t> iterator_reg;

				if (auto it = value2mem_map.find(iterator); it != value2mem_map.end()) {
					if (std::holds_alternative<Reg>(it->second)) {
						iterator_reg = std::get<Reg>(it->second).idx;
						logger->debug("Found iterator in r{}", *iterator_reg);
					}
				}

				// Also check block argument mappings
				if (!iterator_reg.has_value()) {
					if (auto it = live_interval_analysis.block_input_mappings.find(iterator);
						it != live_interval_analysis.block_input_mappings.end()) {
						for (const auto &mapped : it->second) {
							if (auto reg_it = value2mem_map.find(mapped);
								reg_it != value2mem_map.end()) {
								if (std::holds_alternative<Reg>(reg_it->second)) {
									iterator_reg = std::get<Reg>(reg_it->second).idx;
									logger->debug(
										"Found iterator via block mapping in r{}", *iterator_reg);
									break;
								}
							}
						}
					}
				}

				if (iterator_reg.has_value()) {
					available.set(*iterator_reg, false);
					logger->info(
						"ForIter loop variable CANNOT use r{} (iterator register)", *iterator_reg);
				} else {
					logger->error("ForIter iterator register not found - BUG NOT FIXED!");
					// This is a critical error - the iterator must be allocated before the loop
					// variable
				}
			}
		}
	}

	/**
	 * Allocate a register for the current interval
	 */
	void allocate_register(const LiveIntervalAnalysis::LiveInterval &cur,
		std::bitset<32> &free,
		const std::bitset<32> &available,
		mlir::OpBuilder &builder,
		const LiveIntervalAnalysis &live_interval_analysis,
		LiveIntervalSet &active,
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
					logger->debug("Allocated r{} to {}", i, to_string(cur.value));
					break;
				}
			}
		}

		ASSERT(cur_reg.has_value());

		// Handle case where the chosen register is not free (need to save/restore)
		if (!free.test(*cur_reg)) {
			handle_register_conflict(
				cur, *cur_reg, available, builder, live_interval_analysis, logger);
		} else {
			free.set(*cur_reg, false);
		}

		active.insert(cur);

		// CRITICAL FIX FOR FORITER BUG:
		// When allocating a block argument that is a FOR_ITER loop variable, reserve the
		// iterator register for the duration of the loop to prevent it from being reused
		if (std::holds_alternative<mlir::Value>(cur.value)) {
			auto value = std::get<mlir::Value>(cur.value);

			// Check if this is a block argument
			if (mlir::isa<mlir::BlockArgument>(value)) {
				logger->debug("Allocated block argument: {}", to_string(cur.value));

				// Check if this block argument comes from a FOR_ITER
				if (auto it = live_interval_analysis.block_input_mappings.find(cur.value);
					it != live_interval_analysis.block_input_mappings.end()) {

					for (const auto &input : it->second) {
						if (std::holds_alternative<ForwardedOutput>(input)) {
							auto forwarded = std::get<ForwardedOutput>(input);

							if (auto for_iter = mlir::dyn_cast<mlir::emitpybytecode::ForIter>(forwarded.first)) {
								logger->debug("Block argument is FOR_ITER loop variable");
								auto iterator = for_iter.getIterator();
								logger->debug("Iterator value: {}", to_string(iterator));

								// Find the iterator's register
								if (auto iter_it = value2mem_map.find(iterator); iter_it != value2mem_map.end()) {
									if (std::holds_alternative<Reg>(iter_it->second)) {
										auto iterator_reg = std::get<Reg>(iter_it->second).idx;
										logger->debug("Found iterator register: r{}", iterator_reg);

										// Reserve this register for the duration of the loop variable's lifetime
										foriter_reserved_regs[cur.value] = iterator_reg;

										// Mark the iterator register as busy
										free.set(iterator_reg, false);

										logger->info("FOR_ITER FIX: Reserved r{} (iterator) for loop variable {}",
											iterator_reg, to_string(cur.value));
									} else {
										logger->warn("Iterator register is not a Reg!");
									}
								} else {
									logger->error("Iterator not found in value2mem_map!");
								}
							}
						}
					}
				}
			}
		}
	}

	/**
	 * Handle the case where we need a register that's currently in use
	 * by inserting save/restore code
	 */
	void handle_register_conflict(const LiveIntervalAnalysis::LiveInterval &cur,
		size_t cur_reg,
		const std::bitset<32> &available,
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
			if (current_value.isa<mlir::BlockArgument>()) {
				if (auto it = live_interval_analysis.block_input_mappings.find(cur.value);
					it != live_interval_analysis.block_input_mappings.end()) {
					for (auto mapped_value : it->second) {
						ASSERT(!std::holds_alternative<ForwardedOutput>(mapped_value));
						if (clobbers_r0(std::get<mlir::Value>(mapped_value))) {
							ASSERT(current_value.isa<mlir::BlockArgument>());
							current_value = std::get<mlir::Value>(mapped_value);
							break;
						}
					}
				}
			}

			ASSERT(!current_value.isa<mlir::BlockArgument>());
			auto loc = current_value.getLoc();

			// Insert: push r{cur_reg}, move r{scratch}, r{cur_reg}, pop r{cur_reg}
			builder.setInsertionPoint(current_value.getDefiningOp());
			builder.create<mlir::emitpybytecode::Push>(loc, cur_reg);
			builder.setInsertionPointAfter(current_value.getDefiningOp());
			builder.create<mlir::emitpybytecode::Move>(loc, *scratch_reg, cur_reg);
			builder.create<mlir::emitpybytecode::Pop>(loc, cur_reg);

			value2mem_map.insert_or_assign(cur.value, Reg{ .idx = *scratch_reg });

			logger->debug("Register conflict: moved {} from r{} to r{} (scratch)",
				to_string(current_value),
				cur_reg,
				*scratch_reg);
		}
	}

	/**
	 * Free a register when an interval expires
	 */
	void free_register(const LiveIntervalAnalysis::LiveInterval &interval,
		std::bitset<32> &free,
		std::shared_ptr<spdlog::logger> &logger)
	{
		const auto reg = value2mem_map.at(interval.value);
		ASSERT(std::holds_alternative<Reg>(reg));
		size_t reg_idx = std::get<Reg>(reg).idx;
		free.set(reg_idx, true);
		logger->trace("Freed r{} from {}", reg_idx, to_string(interval.value));

		// If this was a FOR_ITER loop variable with a reserved iterator register, free it too
		if (auto it = foriter_reserved_regs.find(interval.value); it != foriter_reserved_regs.end()) {
			auto reserved_reg = it->second;
			free.set(reserved_reg, true);
			logger->info("FOR_ITER FIX: Freed reserved iterator register r{}", reserved_reg);
			foriter_reserved_regs.erase(it);
		}
	}

	/**
	 * Mark a register as used when an interval becomes active
	 */
	void mark_register_used(const LiveIntervalAnalysis::LiveInterval &interval,
		std::bitset<32> &free,
		std::shared_ptr<spdlog::logger> &logger)
	{
		const auto reg = value2mem_map.at(interval.value);
		ASSERT(std::holds_alternative<Reg>(reg));
		size_t reg_idx = std::get<Reg>(reg).idx;
		ASSERT(free.test(reg_idx));
		free.set(reg_idx, false);
		logger->trace("Marked r{} as used for {}", reg_idx, to_string(interval.value));

		// If this is a FOR_ITER loop variable, also mark the iterator register as used
		if (auto it = foriter_reserved_regs.find(interval.value); it != foriter_reserved_regs.end()) {
			auto reserved_reg = it->second;
			free.set(reserved_reg, false);
			logger->trace("FOR_ITER FIX: Marked reserved iterator register r{} as used", reserved_reg);
		}
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
					logger->debug("  {} -> r{}", to_string(value), std::get<Reg>(location).idx);
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
		for (size_t t = 0; t < max_timestep; ++t) {
			llvm::outs() << llvm::format("%3d", t);
		}
		llvm::outs() << "\n";

		// Print separator
		llvm::outs() << "--------------------------------------------------------+-";
		for (size_t t = 0; t < max_timestep; ++t) {
			llvm::outs() << "---";
		}
		llvm::outs() << "\n";

		// Print each value's liveness
		for (const auto &interval : live_interval_analysis->sorted_live_intervals) {
			// Print value name (truncate to 55 chars)
			std::string value_str = to_string(interval.value);
			if (value_str.length() > 55) {
				value_str = value_str.substr(0, 52) + "...";
			}
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
		for (size_t t = 0; t < max_timestep; ++t) {
			llvm::outs() << llvm::format("%3d", t);
		}
		llvm::outs() << "\n";

		// Print separator
		llvm::outs() << "--------------------------------------------------------+-";
		for (size_t t = 0; t < max_timestep; ++t) {
			llvm::outs() << "---";
		}
		llvm::outs() << "\n";

		// Print each value's register assignment
		for (const auto &interval : live_interval_analysis->sorted_live_intervals) {
			// Print value name (truncate to 55 chars)
			std::string value_str = to_string(interval.value);
			if (value_str.length() > 55) {
				value_str = value_str.substr(0, 52) + "...";
			}
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
