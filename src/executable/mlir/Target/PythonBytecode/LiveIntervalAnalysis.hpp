#pragma once

#include "LiveAnalysis.hpp"
#include "RegisterAllocationLogger.hpp"
#include "RegisterAllocationTypes.hpp"

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

namespace codegen {

/**
 * LiveIntervalAnalysis builds on LiveAnalysis to compute live intervals for each value.
 * A live interval is a set of program points (timesteps) where a value is alive.
 *
 * This information is used by the register allocator to determine when registers
 * can be reused.
 */
class LiveIntervalAnalysis
{
  public:
	/**
	 * Represents the live interval for a single value
	 */
	struct LiveInterval
	{
		// Each interval is [start, end) - half-open range
		using Interval = std::tuple<size_t, size_t>;

		// Multiple intervals for the same value (for values that go dead and then alive again)
		std::vector<Interval> intervals;

		// The value this interval represents
		std::variant<mlir::Value, ForwardedOutput> value;

		// Start of the first interval
		size_t start() const { return std::get<0>(intervals.front()); }

		// End of the last interval
		size_t end() const { return std::get<1>(intervals.back()); }

		/**
		 * Check if this interval is alive at the given position.
		 * Uses precise sub-interval membership rather than a conservative span check.
		 * GET_ITER intervals are collapsed to a single contiguous span by
		 * extend_iterator_liveness() so they remain alive throughout the loop.
		 */
		bool alive_at(size_t pos) const
		{
			return std::find_if(intervals.begin(),
					   intervals.end(),
					   [pos](const Interval &interval) {
						   auto [start, end] = interval;
						   return pos >= start && pos < end;
					   })
				   != intervals.end();
		}

		/**
		 * Check if this interval overlaps with another.
		 * Intervals are half-open [start, end), so [a,b) and [c,d) overlap iff a < d && c < b.
		 */
		bool overlaps(const LiveInterval &other) const
		{
			for (const auto &[start, end] : intervals) {
				for (const auto &[other_start, other_end] : other.intervals) {
					if (other_start < end && start < other_end) { return true; }
				}
			}
			return false;
		}
	};

	// Live intervals sorted by start position
	std::vector<LiveInterval> sorted_live_intervals;

	// Maps values to block arguments they flow into (inverted from LiveAnalysis)
	ValueMapping<std::vector<std::variant<mlir::Value, ForwardedOutput>>> block_input_mappings;

	/**
	 * Analyze the function to compute live intervals
	 */
	void analyse(mlir::func::FuncOp &func)
	{
		auto logger = get_regalloc_logger();
		logger->info("Starting live interval analysis");

		// First run live analysis
		LiveAnalysis live_analysis{};
		live_analysis.analyse(func);

		// Invert the block_input_mappings for easier lookup
		for (auto [key, value] : live_analysis.block_input_mappings) {
			for (const auto &el : value) { block_input_mappings[el].push_back(key); }
		}

		// Build live intervals from liveness information.
		// An index map provides O(log n) lookup instead of O(n) linear scan per value.
		std::vector<LiveInterval> unsorted_live_intervals;
		std::map<std::variant<mlir::Value, ForwardedOutput>, size_t, ValueMappingComparator>
			interval_index;

		for (size_t timestep = 0; const auto &alive_values : live_analysis.alive_at_timestep) {
			for (const auto &alive_value : alive_values) {
				update_interval(alive_value, timestep, unsorted_live_intervals, interval_index);
			}
			timestep++;
		}

		// Sort by start position for linear scan register allocation
		std::sort(unsorted_live_intervals.begin(),
			unsorted_live_intervals.end(),
			[](const LiveInterval &lhs, const LiveInterval &rhs) {
				return lhs.start() < rhs.start();
			});

		sorted_live_intervals = std::move(unsorted_live_intervals);

		// Collapse GET_ITER live intervals to contiguous spans.
		// This ensures that iterator values stay permanently active throughout the loop,
		// eliminating the need for inactive→active reload logic and simplifying the allocator.
		extend_iterator_liveness();

		logger->info(
			"Live interval analysis complete. Found {} intervals", sorted_live_intervals.size());

		for (const auto &interval : sorted_live_intervals) {
			logger->trace("LiveInterval for {}: start={}, end={}",
				interval.value,
				interval.start(),
				interval.end());
		}
	}

  private:
	/**
	 * Collapse GET_ITER live intervals to a single contiguous span.
	 *
	 * Backward dataflow may produce gaps in a GET_ITER value's live interval when loop
	 * back-edges cause the iterator to appear as a separate sub-interval on each iteration.
	 * With the now-precise alive_at() check, such gaps would allow the allocator to move
	 * the iterator to inactive and potentially clobber its register.
	 *
	 * By collapsing [start, gap, end] → [start, end), the iterator remains permanently
	 * active throughout the loop. This is semantically correct because the iterator MUST
	 * survive for the entire duration of the FOR loop.
	 */
	void extend_iterator_liveness()
	{
		for (auto &interval : sorted_live_intervals) {
			if (!std::holds_alternative<mlir::Value>(interval.value)) { continue; }
			auto val = std::get<mlir::Value>(interval.value);
			if (!val.getDefiningOp()) { continue; }
			if (!mlir::isa<mlir::emitpybytecode::GetIter>(val.getDefiningOp())) { continue; }

			// Collapse all sub-intervals into one contiguous [start, end) span
			const size_t first = interval.start();
			const size_t last = interval.end();
			interval.intervals.clear();
			interval.intervals.emplace_back(first, last);
		}
	}

	/**
	 * Update or create a live interval for the given value at the current timestep.
	 *
	 * Uses an index map for O(log n) lookup rather than a linear scan over all intervals.
	 */
	void update_interval(
		const std::variant<mlir::Value, ForwardedOutput, LiveAnalysis::BlockArgumentInputs>
			&alive_value,
		size_t timestep,
		std::vector<LiveInterval> &intervals,
		std::map<std::variant<mlir::Value, ForwardedOutput>, size_t, ValueMappingComparator>
			&interval_index)
	{
		// Extract the actual values to track from alive_value
		std::vector<std::variant<mlir::Value, ForwardedOutput>> values_to_track;

		if (std::holds_alternative<mlir::Value>(alive_value)) {
			values_to_track.push_back(std::get<mlir::Value>(alive_value));
		} else if (std::holds_alternative<ForwardedOutput>(alive_value)) {
			values_to_track.push_back(std::get<ForwardedOutput>(alive_value));
		} else {
			// BlockArgumentInputs: track all the inputs
			const auto &inputs =
				std::get<1>(std::get<LiveAnalysis::BlockArgumentInputs>(alive_value));
			values_to_track.insert(values_to_track.end(), inputs.begin(), inputs.end());
		}

		// Update interval for each value using the O(log n) index map
		for (auto value : values_to_track) {
			auto idx_it = interval_index.find(value);

			if (idx_it == interval_index.end()) {
				// Create new interval and record its index
				const size_t idx = intervals.size();
				intervals.emplace_back(
					std::vector{ std::make_tuple(timestep, timestep + 1) }, value);
				interval_index.emplace(value, idx);
			} else {
				// Extend existing interval
				auto &value_intervals = intervals[idx_it->second].intervals;
				const size_t end = std::get<1>(value_intervals.back());

				if (timestep == end) {
					// Consecutive timestep: extend the current interval
					std::get<1>(value_intervals.back())++;
				} else {
					// Gap detected: start a new sub-interval
					value_intervals.emplace_back(timestep, timestep + 1);
				}
			}
		}
	}
};

}// namespace codegen
