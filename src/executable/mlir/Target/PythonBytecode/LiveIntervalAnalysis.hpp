#pragma once

#include "LiveAnalysis.hpp"
#include "RegisterAllocationLogger.hpp"
#include "RegisterAllocationTypes.hpp"

#include <algorithm>
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
		 * Check if this interval is alive at the given position
		 */
		bool alive_at(size_t pos) const
		{
			// FIXME: the commented code is correct, but currently there is no logic
			//        to populate a register when an interval goes from inactive to active
			//        (i.e., the register is potentially clobbered)
			// return std::find_if(intervals.begin(),
			// 		   intervals.end(),
			// 		   [pos](const Interval &interval) {
			// 			   auto [start, end] = interval;
			// 			   return pos >= start && pos < end;
			// 		   })
			// 	   != intervals.end();

			// Conservative approximation: check only the full span
			return pos >= start() && pos < end();
		}

		/**
		 * Check if this interval overlaps with another
		 */
		bool overlaps(const LiveInterval &other) const
		{
			// Naive quadratic search - could be optimized with interval tree
			for (const auto &[start, end] : intervals) {
				for (const auto &[other_start, other_end] : other.intervals) {
					if (other_start >= start && other_start <= end) { return true; }
					if (other_end >= start && other_end <= end) { return true; }
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

		// Build live intervals from liveness information
		std::vector<LiveInterval> unsorted_live_intervals;

		for (size_t timestep = 0; const auto &alive_values : live_analysis.alive_at_timestep) {
			for (const auto &alive_value : alive_values) {
				update_interval(alive_value, timestep, unsorted_live_intervals);
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

		logger->info("Live interval analysis complete. Found {} intervals",
			sorted_live_intervals.size());

		// Log intervals at debug level (temporarily for debugging)
		for (const auto &interval : sorted_live_intervals) {
			// Log all intervals, especially GET_ITER
			if (std::holds_alternative<mlir::Value>(interval.value)) {
				auto val = std::get<mlir::Value>(interval.value);
				if (val.getDefiningOp() && mlir::isa<mlir::emitpybytecode::GetIter>(val.getDefiningOp())) {
					logger->info("GET_ITER LiveInterval: start={}, end={}, {} sub-intervals",
						interval.start(),
						interval.end(),
						interval.intervals.size());
					for (size_t i = 0; i < interval.intervals.size(); ++i) {
						auto [s, e] = interval.intervals[i];
						logger->info("  Interval {}: [{}, {})", i, s, e);
					}
				}
			}
			logger->trace("LiveInterval for {}: start={}, end={}",
				to_string(interval.value),
				interval.start(),
				interval.end());
		}
	}

  private:
	/**
	 * Update or create a live interval for the given value at the current timestep
	 */
	void update_interval(
		const std::variant<mlir::Value, ForwardedOutput, LiveAnalysis::BlockArgumentInputs>
			&alive_value,
		size_t timestep,
		std::vector<LiveInterval> &intervals)
	{
		// Extract the actual values to track from alive_value
		std::vector<std::variant<mlir::Value, ForwardedOutput>> values_to_track;

		if (std::holds_alternative<mlir::Value>(alive_value)) {
			values_to_track.push_back(std::get<mlir::Value>(alive_value));
		} else if (std::holds_alternative<ForwardedOutput>(alive_value)) {
			values_to_track.push_back(std::get<ForwardedOutput>(alive_value));
		} else {
			// BlockArgumentInputs: track all the inputs
			const auto &inputs = std::get<1>(std::get<LiveAnalysis::BlockArgumentInputs>(alive_value));
			values_to_track.insert(values_to_track.end(), inputs.begin(), inputs.end());
		}

		// Update interval for each value
		for (auto value : values_to_track) {
			auto it = std::find_if(intervals.begin(), intervals.end(), [&value](const auto &interval) {
				return interval.value == value;
			});

			if (it == intervals.end()) {
				// Create new interval
				intervals.emplace_back(std::vector{ std::make_tuple(timestep, timestep + 1) }, value);
			} else {
				// Extend existing interval
				auto &value_intervals = it->intervals;
				const size_t end = std::get<1>(value_intervals.back());

				if (timestep == end) {
					// Consecutive timestep: extend the current interval
					std::get<1>(value_intervals.back())++;
				} else {
					// Gap detected: start a new interval
					value_intervals.emplace_back(timestep, timestep + 1);
				}
			}
		}
	}
};

}// namespace codegen
