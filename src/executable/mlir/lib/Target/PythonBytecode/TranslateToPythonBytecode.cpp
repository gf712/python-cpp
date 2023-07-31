#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/PythonOps.hpp"
#include "Target/PythonBytecode/PythonBytecodeEmitter.hpp"
#include "executable/Mangler.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "executable/bytecode/instructions/BinaryOperation.hpp"
#include "executable/bytecode/instructions/BinarySubscript.hpp"
#include "executable/bytecode/instructions/BuildDict.hpp"
#include "executable/bytecode/instructions/BuildList.hpp"
#include "executable/bytecode/instructions/BuildSet.hpp"
#include "executable/bytecode/instructions/BuildSlice.hpp"
#include "executable/bytecode/instructions/BuildTuple.hpp"
#include "executable/bytecode/instructions/ClearExceptionState.hpp"
#include "executable/bytecode/instructions/CompareOperation.hpp"
#include "executable/bytecode/instructions/DeleteFast.hpp"
#include "executable/bytecode/instructions/DeleteGlobal.hpp"
#include "executable/bytecode/instructions/DeleteName.hpp"
#include "executable/bytecode/instructions/DeleteSubscript.hpp"
#include "executable/bytecode/instructions/DictAdd.hpp"
#include "executable/bytecode/instructions/DictUpdate.hpp"
#include "executable/bytecode/instructions/ForIter.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "executable/bytecode/instructions/FunctionCallEx.hpp"
#include "executable/bytecode/instructions/FunctionCallWithKeywords.hpp"
#include "executable/bytecode/instructions/GetIter.hpp"
#include "executable/bytecode/instructions/GetYieldFromIter.hpp"
#include "executable/bytecode/instructions/ImportFrom.hpp"
#include "executable/bytecode/instructions/ImportName.hpp"
#include "executable/bytecode/instructions/ImportStar.hpp"
#include "executable/bytecode/instructions/InplaceOp.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "executable/bytecode/instructions/Jump.hpp"
#include "executable/bytecode/instructions/JumpIfExceptionMatch.hpp"
#include "executable/bytecode/instructions/JumpIfFalse.hpp"
#include "executable/bytecode/instructions/JumpIfNotExceptionMatch.hpp"
#include "executable/bytecode/instructions/JumpIfTrue.hpp"
#include "executable/bytecode/instructions/LeaveExceptionHandling.hpp"
#include "executable/bytecode/instructions/ListAppend.hpp"
#include "executable/bytecode/instructions/ListExtend.hpp"
#include "executable/bytecode/instructions/ListToTuple.hpp"
#include "executable/bytecode/instructions/LoadAssertionError.hpp"
#include "executable/bytecode/instructions/LoadAttr.hpp"
#include "executable/bytecode/instructions/LoadBuildClass.hpp"
#include "executable/bytecode/instructions/LoadClosure.hpp"
#include "executable/bytecode/instructions/LoadConst.hpp"
#include "executable/bytecode/instructions/LoadDeref.hpp"
#include "executable/bytecode/instructions/LoadFast.hpp"
#include "executable/bytecode/instructions/LoadGlobal.hpp"
#include "executable/bytecode/instructions/LoadMethod.hpp"
#include "executable/bytecode/instructions/LoadName.hpp"
#include "executable/bytecode/instructions/MakeFunction.hpp"
#include "executable/bytecode/instructions/Move.hpp"
#include "executable/bytecode/instructions/Pop.hpp"
#include "executable/bytecode/instructions/Push.hpp"
#include "executable/bytecode/instructions/RaiseVarargs.hpp"
#include "executable/bytecode/instructions/ReRaise.hpp"
#include "executable/bytecode/instructions/ReturnValue.hpp"
#include "executable/bytecode/instructions/SetAdd.hpp"
#include "executable/bytecode/instructions/SetUpdate.hpp"
#include "executable/bytecode/instructions/SetupExceptionHandling.hpp"
#include "executable/bytecode/instructions/SetupWith.hpp"
#include "executable/bytecode/instructions/StoreAttr.hpp"
#include "executable/bytecode/instructions/StoreDeref.hpp"
#include "executable/bytecode/instructions/StoreFast.hpp"
#include "executable/bytecode/instructions/StoreGlobal.hpp"
#include "executable/bytecode/instructions/StoreName.hpp"
#include "executable/bytecode/instructions/StoreSubscript.hpp"
#include "executable/bytecode/instructions/ToBool.hpp"
#include "executable/bytecode/instructions/Unary.hpp"
#include "executable/bytecode/instructions/UnpackSequence.hpp"
#include "executable/bytecode/instructions/WithExceptStart.hpp"
#include "executable/bytecode/instructions/YieldFrom.hpp"
#include "executable/bytecode/instructions/YieldLoad.hpp"
#include "executable/bytecode/instructions/YieldValue.hpp"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/TypeSwitch.h"

#include <map>
#include <optional>
#include <ranges>
#include <set>

using namespace mlir;

namespace codegen {

namespace {
	bool is_function_call(mlir::Value value)
	{
		return mlir::isa<mlir::emitpybytecode::FunctionCallOp>(value.getDefiningOp())
			   || mlir::isa<mlir::emitpybytecode::FunctionCallExOp>(value.getDefiningOp())
			   || mlir::isa<mlir::emitpybytecode::FunctionCallWithKeywordsOp>(
				   value.getDefiningOp());
	}

	std::vector<mlir::Block *> sortBlocks(mlir::Region &region)
	{
		mlir::SetVector<Block *> blocks;

		for (auto &b : region) {
			if (blocks.count(&b) == 0) {
				for (llvm::scc_iterator<mlir::Block *> it = llvm::scc_begin(&b),
													   end = llvm::scc_end(&b);
					 it != end;
					 ++it) {
					blocks.insert(it->begin(), it->end());
				}
			}
		}

		ASSERT(blocks.size() == region.getBlocks().size());

		return std::vector(blocks.rbegin(), blocks.rend());
	}
}// namespace

struct LiveAnalysis
{
	using BlockArgumentInputs = std::tuple<mlir::BlockArgument, std::vector<mlir::Value>>;
	using AliveAtTimestepT = std::vector<std::variant<mlir::Value, BlockArgumentInputs>>;
	std::vector<AliveAtTimestepT> alive_at_timestep;

	std::map<Value, Value, decltype([](const mlir::Value &lhs, const mlir::Value &rhs) {
		return lhs.getImpl() < rhs.getImpl();
	})>
		block_input_mappings;

	void analyse(mlir::func::FuncOp &fn)
	{
		auto &region = fn.getRegion();

		auto sorted_blocks = sortBlocks(region);

		auto add_value = [](mlir::Value value, AliveAtTimestepT &alive) {
			auto it = std::find_if(alive.begin(), alive.end(), [&value](const auto &el) {
				ASSERT(std::holds_alternative<Value>(el));
				return std::get<Value>(el) == value;
			});
			if (it == alive.end()) { alive.push_back(std::move(value)); }
		};

		AsmState state{ fn.getOperation() };
		std::vector<std::pair<mlir::Value, mlir::BlockArgument>> block_parameters_to_args;
		std::map<Block *, std::pair<size_t, size_t>> blocks_span;
		for (auto *block : sorted_blocks) {
			const auto start = alive_at_timestep.size();
			if (!sortTopologically(block)) { std::abort(); }
			// block->print(llvm::outs(), state);
			for (auto &op : block->getOperations()) {
				auto &alive = alive_at_timestep.emplace_back();
				// ASSERT(op.getOpResults().size() <= 1);
				for (const auto &result : op.getResults()) { add_value(result, alive); }
				for (const auto &operand : op.getOperands()) { add_value(operand, alive); }
				// if (!op.getResults().empty()) {
				// 	llvm::outs() << "@" << (void *)Value{ op.getResults().back() }.getImpl()
				// 				 << ": ";
				// 	op.print(llvm::outs(), state);
				// 	llvm::outs() << '\n';
				// }
			}
			if (block->getTerminator()) {
				if (auto branch =
						dyn_cast<mlir::emitpybytecode::JumpIfFalse>(block->getTerminator())) {
					auto *true_block = branch.getTrueDest();
					ASSERT(
						branch.getTrueDestOperands().size() == true_block->getArguments().size());
					for (const auto &[p, arg] :
						llvm::zip(branch.getTrueDestOperands(), true_block->getArguments())) {
						block_parameters_to_args.emplace_back(p, arg);
					}
					auto *false_block = branch.getFalseDest();
					ASSERT(
						branch.getFalseDestOperands().size() == false_block->getArguments().size());
					for (const auto &[p, arg] :
						llvm::zip(branch.getFalseDestOperands(), false_block->getArguments())) {
						block_parameters_to_args.emplace_back(p, arg);
					}
				} else if (auto branch = dyn_cast<mlir::cf::BranchOp>(block->getTerminator())) {
					auto *jmp_block = branch.getDest();
					ASSERT(branch.getDestOperands().size() == jmp_block->getArguments().size());
					for (const auto &[p, arg] :
						llvm::zip(branch.getDestOperands(), jmp_block->getArguments())) {
						block_parameters_to_args.emplace_back(p, arg);
					}
				} else if (auto for_iter =
							   dyn_cast<mlir::emitpybytecode::ForIter>(block->getTerminator())) {
					ASSERT(for_iter.getBody()->getArguments().size() == 2);
					block_parameters_to_args.emplace_back(
						for_iter.getValue(), for_iter.getBody()->getArgument(0));
					block_parameters_to_args.emplace_back(
						for_iter.getIterator(), for_iter.getBody()->getArgument(1));
				}
			}
			blocks_span.emplace(block, std::pair{ start, alive_at_timestep.size() });
		}

		for (const auto &[param, arg] : block_parameters_to_args) {
			auto *bb = arg.getOwner();
			const auto [start, end] = blocks_span.at(bb);
			auto block_timesteps =
				std::span{ alive_at_timestep.begin() + start, alive_at_timestep.begin() + end };
			for (auto &ts : block_timesteps) {
				for (auto &val : ts) {
					if (std::holds_alternative<mlir::Value>(val)
						&& std::get<mlir::Value>(val).isa<BlockArgument>()
						&& std::get<mlir::Value>(val).cast<BlockArgument>() == arg) {
						val = BlockArgumentInputs{ arg, { param } };
						block_input_mappings[param] = arg;
					} else if (std::holds_alternative<BlockArgumentInputs>(val)
							   && std::get<0>(std::get<BlockArgumentInputs>(val)) == arg) {
						std::get<1>(std::get<BlockArgumentInputs>(val)).push_back(param);
						block_input_mappings[param] = arg;
					}
				}
			}
		}

		for (auto &values : alive_at_timestep | std::ranges::views::reverse) {
			for (auto &value : values | std::ranges::views::reverse) {
				if (std::holds_alternative<BlockArgumentInputs>(value)) {
					value = std::get<0>(std::get<BlockArgumentInputs>(value));
				}
				auto start = block_input_mappings.find(std::get<Value>(value));
				auto it = start;
				while (it != block_input_mappings.end()) {
					value = it->second;
					start->second = std::get<Value>(value);
					it = block_input_mappings.find(std::get<Value>(value));
				}
			}
		}

		// for (size_t idx = 0; const auto &values : alive_at_timestep) {
		// 	llvm::outs() << idx++ << ": ";
		// 	for (auto value : values) {
		// 		std::visit(overloaded{
		// 					   [](Value v) {
		// 						   llvm::outs() << v.getImpl() << '[';
		// 						   v.print(llvm::outs());
		// 						   llvm::outs() << "], ";
		// 					   },
		// 					   [](const BlockArgumentInputs &b) {
		// 						   llvm::outs()
		// 							   << static_cast<Value>(std::get<0>(b)).getImpl() << '{';
		// 						   for (const auto &v : std::get<1>(b)) {
		// 							   llvm::outs() << v.getImpl() << ", ";
		// 						   }
		// 						   llvm::outs() << "}, ";
		// 					   },
		// 				   },
		// 			value);
		// 	}
		// 	llvm::outs() << '\n';
		// }
	}
};

struct LiveIntervalAnalysis
{
	struct LiveInterval
	{
		// start, end
		using Interval = std::tuple<size_t, size_t>;
		std::vector<Interval> intervals;
		mlir::Value value;

		size_t start() const { return std::get<0>(intervals.front()); }

		size_t end() const { return std::get<1>(intervals.back()); }

		bool alive_at(size_t pos) const
		{
			// FIXME: the commented code is correct, but currently there is no logic
			//        to populate a register when an interval goes from inactive to active
			//        ie. the register is potentially clobbered
			// return std::find_if(intervals.begin(),
			// 		   intervals.end(),
			// 		   [pos](const Interval &interval) {
			// 			   auto [start, end] = interval;
			// 			   return pos >= start && pos < end;
			// 		   })
			// 	   != intervals.end();
			return pos >= start() && pos < end();
		}

		bool overlaps(const LiveInterval &other) const
		{
			// naive quadratic search
			for (const auto &[start, end] : intervals) {
				for (const auto &[other_start, other_end] : other.intervals) {
					if (other_start >= start && other_start <= end) { return true; }
					if (other_end >= start && other_end <= end) { return true; }
				}
			}

			return false;
		}
	};

	std::vector<LiveInterval> sorted_live_intervals;
	std::
		map<Value, std::vector<Value>, decltype([](const mlir::Value &lhs, const mlir::Value &rhs) {
			return lhs.getImpl() < rhs.getImpl();
		})>
			block_input_mappings;

	void analyse(mlir::func::FuncOp &func)
	{
		LiveAnalysis live_analysis{};
		live_analysis.analyse(func);
		for (auto [key, value] : live_analysis.block_input_mappings) {
			block_input_mappings[value].push_back(key);
		}

		std::vector<LiveInterval> unsorted_live_intervals;
		auto update_interval =
			[this, &unsorted_live_intervals](
				const std::variant<Value, LiveAnalysis::BlockArgumentInputs> &inputs,
				size_t current) {
				std::vector<Value> compute_live_interval_values;
				if (std::holds_alternative<Value>(inputs)) {
					compute_live_interval_values.push_back(std::get<Value>(inputs));
				} else {
					compute_live_interval_values.insert(compute_live_interval_values.end(),
						std::get<1>(std::get<LiveAnalysis::BlockArgumentInputs>(inputs)).begin(),
						std::get<1>(std::get<LiveAnalysis::BlockArgumentInputs>(inputs)).end());
				}

				for (auto value : compute_live_interval_values) {
					if (auto it = std::find_if(unsorted_live_intervals.begin(),
							unsorted_live_intervals.end(),
							[&value](const auto &el) { return el.value == value; });
						it == unsorted_live_intervals.end()) {
						unsorted_live_intervals.emplace_back(
							std::vector{ std::make_tuple(current, current + 1) }, value);
					} else {
						auto &intervals = it->intervals;
						const size_t end = std::get<1>(intervals.back());
						if (current == end) {
							std::get<1>(intervals.back())++;
						} else {
							intervals.emplace_back(current, current + 1);
						}
					}
				}
			};

		for (size_t i = 0; const auto &vals : live_analysis.alive_at_timestep) {
			for (const auto &val : vals) { update_interval(val, i); }
			i++;
		}

		std::sort(unsorted_live_intervals.begin(),
			unsorted_live_intervals.end(),
			[](const LiveIntervalAnalysis::LiveInterval &lhs,
				const LiveIntervalAnalysis::LiveInterval &rhs) {
				return lhs.start() < rhs.start();
			});
		sorted_live_intervals = std::move(unsorted_live_intervals);

		// for (const auto &live_interval : sorted_live_intervals) {
		// 	auto [intervals, value] = live_interval;
		// 	llvm::outs() << static_cast<const void *>(value.getImpl()) << " ";
		// 	for (const auto &interval : intervals) {
		// 		auto [start, end] = interval;
		// 		llvm::outs() << fmt::format("[{}, {}[ ", start, end);
		// 	}
		// 	llvm::outs() << '\n';
		// }
	}
};

struct LinearScanRegisterAllocation
{
	struct Reg
	{
		size_t idx;
	};
	struct StackLocation
	{
		size_t idx;
	};
	using ValueLocation = std::variant<Reg, StackLocation>;
	std::map<mlir::Value,
		ValueLocation,
		decltype([](const mlir::Value &lhs, const mlir::Value &rhs) {
			return lhs.getImpl() < rhs.getImpl();
		})>
		value2mem_map;

	void analyse(mlir::func::FuncOp &func, mlir::OpBuilder builder)
	{
		LiveIntervalAnalysis live_interval_analysis;
		live_interval_analysis.analyse(func);

		auto unhandled = std::span(live_interval_analysis.sorted_live_intervals.begin(),
			live_interval_analysis.sorted_live_intervals.end());
		ASSERT(std::is_sorted(unhandled.begin(),
			unhandled.end(),
			[](const LiveIntervalAnalysis::LiveInterval &lhs,
				const LiveIntervalAnalysis::LiveInterval &rhs) {
				return lhs.start() < rhs.start();
			}));

		auto increasing_endpoint_cmp = [](const LiveIntervalAnalysis::LiveInterval &lhs,
										   const LiveIntervalAnalysis::LiveInterval &rhs) {
			return lhs.end() < rhs.end();
		};

		std::multiset<LiveIntervalAnalysis::LiveInterval, decltype(increasing_endpoint_cmp)> active;
		std::multiset<LiveIntervalAnalysis::LiveInterval, decltype(increasing_endpoint_cmp)>
			inactive;
		std::multiset<LiveIntervalAnalysis::LiveInterval, decltype(increasing_endpoint_cmp)>
			handled;

		std::bitset<32> free;
		free.set();

		for (const auto &interval : unhandled) {
			// the result of a function call is always in Reg{0}, so we start by claiming Reg{0} for
			// the result of all call operations
			if ((interval.value.getDefiningOp() && is_function_call(interval.value))) {
				value2mem_map[interval.value] = Reg{ .idx = 0 };
				inactive.insert(interval);
			}

			// account for block arguments that could be the result of a function call
			if (auto it = live_interval_analysis.block_input_mappings.find(interval.value);
				it != live_interval_analysis.block_input_mappings.end()) {
				for (auto mapped_value : it->second) {
					if ((mapped_value.getDefiningOp() && is_function_call(mapped_value))) {
						value2mem_map[interval.value] = Reg{ .idx = 0 };
						inactive.insert(interval);
						break;
					}
				}
			}
		}

		while (!unhandled.empty()) {
			// llvm::outs() << "free: " << free << '\n';
			const auto &cur = *unhandled.begin();
			unhandled = unhandled.subspan(1, unhandled.size() - 1);

			// const_cast<mlir::Value &>(cur.value).print(llvm::outs());
			// llvm::outs() << '\n';

			// check for active intervals that expired
			for (auto it = active.begin(); it != active.end();) {
				const auto &interval = *it;
				ASSERT(interval.value != cur.value);
				if (interval.end() < cur.start()) {
					handled.insert(interval);
					it = active.erase(it);
					const auto reg = value2mem_map.at(interval.value);
					ASSERT(std::holds_alternative<Reg>(reg));
					free.set(std::get<Reg>(reg).idx, true);
				} else if (!interval.alive_at(cur.start())) {
					inactive.insert(interval);
					it = active.erase(it);
					const auto reg = value2mem_map.at(interval.value);
					ASSERT(std::holds_alternative<Reg>(reg));
					free.set(std::get<Reg>(reg).idx, true);
				} else {
					++it;
				}
			}
			// check for inactive intervals that expired or become reactivated
			for (auto it = inactive.begin(); it != inactive.end();) {
				const auto &interval = *it;
				if (interval.value == cur.value) {
					ASSERT((interval.value.getDefiningOp() && is_function_call(interval.value))
						   || (live_interval_analysis.block_input_mappings.contains(interval.value)
							   && std::ranges::any_of(
								   live_interval_analysis.block_input_mappings.find(interval.value)
									   ->second,
								   [](auto mapped_value) {
									   return mapped_value.getDefiningOp()
											  && is_function_call(mapped_value);
								   })));
					active.insert(interval);
					it = inactive.erase(it);
				} else if (interval.end() < cur.start()) {
					handled.insert(interval);
					it = inactive.erase(it);
				} else if (interval.alive_at(cur.start())) {
					active.insert(interval);
					const auto reg = value2mem_map.at(interval.value);
					ASSERT(std::holds_alternative<Reg>(reg));
					ASSERT(free.test(std::get<Reg>(reg).idx));
					free.set(std::get<Reg>(reg).idx, false);
				} else {
					++it;
				}
			}

			auto f = free;
			// collect available registers
			auto overlaps =
				std::views::filter([&cur](const auto &interval) { return interval.overlaps(cur); });
			for (const auto &interval : inactive | overlaps) {
				if (auto it = value2mem_map.find(interval.value); it != value2mem_map.end()) {
					const auto reg = it->second;
					ASSERT(std::holds_alternative<Reg>(reg));

					// if it is still inactive it should be ok if this register is still being used
					// we just don't want it to be used when the interval becomes active
					f.set(std::get<Reg>(reg).idx, false);
				}
			}

			for (const auto &interval : unhandled | overlaps) {
				if (auto it = value2mem_map.find(interval.value); it != value2mem_map.end()) {
					const auto reg = it->second;
					ASSERT(std::holds_alternative<Reg>(reg));

					// if it is unhandled it should be ok if this register is still being used
					// we just don't want it to be used when the interval becomes active
					f.set(std::get<Reg>(reg).idx, false);
				}
			}

			if (f.none()) {
				TODO();
			} else {
				std::optional<size_t> cur_reg;
				if (auto it = value2mem_map.find(cur.value); it == value2mem_map.end()) {
					for (size_t i = 0; i < f.size(); ++i) {
						if (f.test(i)) {
							value2mem_map[cur.value] = Reg{ .idx = i };
							cur_reg = i;
							break;
						}
					}
				} else {
					ASSERT(std::holds_alternative<Reg>(it->second));
					cur_reg = std::get<Reg>(it->second).idx;
				}

				ASSERT(cur_reg.has_value());
				// const_cast<mlir::Value &>(cur.value).print(llvm::outs());
				// llvm::outs() << '\n';
				if (!free.test(*cur_reg)) {
					std::optional<size_t> scratch_reg;
					for (size_t i = 1; i < f.size(); ++i) {
						if (f.test(i)) {
							scratch_reg = i;
							break;
						}
					}
					ASSERT(scratch_reg.has_value());
					if (!cur.value.isa<mlir::BlockArgument>()) {
						auto loc = cur.value.getLoc();
						builder.setInsertionPoint(cur.value.getDefiningOp());
						builder.create<mlir::emitpybytecode::Push>(loc, *cur_reg);
						builder.setInsertionPointAfter(cur.value.getDefiningOp());
						builder.create<mlir::emitpybytecode::Move>(loc, *scratch_reg, *cur_reg);
						builder.create<mlir::emitpybytecode::Pop>(loc, *cur_reg);
						value2mem_map[cur.value] = Reg{ .idx = *scratch_reg };
						free.set(*scratch_reg, false);
					}
				} else {
					free.set(*cur_reg, false);
				}
				active.insert(cur);
			}
		}

		decltype(value2mem_map) value2mem_map_additional;
		for (auto [value, reg] : value2mem_map) {
			if (auto it = live_interval_analysis.block_input_mappings.find(value);
				it != live_interval_analysis.block_input_mappings.end()) {
				for (auto mapped_value : it->second) {
					value2mem_map_additional[mapped_value] = reg;
				}
			}
		}
		value2mem_map.merge(std::move(value2mem_map_additional));

		{
			const auto end = std::max_element(live_interval_analysis.sorted_live_intervals.begin(),
				live_interval_analysis.sorted_live_intervals.end(),
				[](const auto &lhs, const auto &rhs) {
					return lhs.end() < rhs.end();
				})->end();

			std::vector<std::vector<void *>> values_by_timestep;
			for (size_t i = 0; i < end; ++i) {
				values_by_timestep.emplace_back(free.size(), nullptr);
			}

			for (const auto &interval : live_interval_analysis.sorted_live_intervals) {
				for (auto [start, end] : interval.intervals) {
					for (; start < end; ++start) {
						auto reg = value2mem_map[interval.value];
						if (std::holds_alternative<Reg>(reg)) {
							values_by_timestep[start][std::get<Reg>(reg).idx] =
								interval.value.getImpl();
						} else {
							TODO();
						}
					}
				}
			}

			// for (size_t i = 0; const auto &ts : values_by_timestep) {
			// 	llvm::outs() << i++ << ' ';
			// 	for (auto *val : ts) { llvm::outs() << val << ' '; }
			// 	llvm::outs() << '\n';
			// }
		}

		// llvm::outs() << "Register allocator modified function:\n";
		// func.print(llvm::outs());
		// llvm::outs() << '\n';
	}
};

struct PythonBytecodeEmitter
{
	struct FunctionInfo
	{
		std::vector<std::unique_ptr<Instruction>> instructions;
		std::vector<::py::Value> m_consts;
		std::vector<std::string> m_varnames;
		std::vector<std::string> m_names;
		std::vector<std::string> m_cellvars;
		std::vector<std::string> m_freevars;
		std::vector<size_t> m_cell2arg;
		size_t m_arg_count;
		size_t m_kwonly_arg_count;
		size_t m_stack_size;
		CodeFlags m_flags = CodeFlags::create();

		void set_varargs() { m_flags.set(CodeFlags::Flag::VARARGS); }

		void set_kwargs() { m_flags.set(CodeFlags::Flag::VARKEYWORDS); }

		void set_is_generator() { m_flags.set(CodeFlags::Flag::GENERATOR); }

		void set_is_class() { m_flags.set(CodeFlags::Flag::CLASS); }

		void set_kwonlyarg_count(size_t kwonlyarg_count) { m_kwonly_arg_count = kwonlyarg_count; }

		size_t add_const(::py::Value value)
		{
			if (auto it = std::find_if(m_consts.begin(),
					m_consts.end(),
					[&value](const auto &el) {
						if (std::holds_alternative<::py::Number>(el)
							&& std::holds_alternative<::py::Number>(value)) {
							return std::get<::py::Number>(el).value.index()
									   == std::get<::py::Number>(value).value.index()
								   && el == value;
						}
						return el.index() == value.index() && el == value;
					});
				it != m_consts.end()) {
				return std::distance(m_consts.begin(), it);
			}
			m_consts.push_back(std::move(value));
			return m_consts.size() - 1;
		}

		size_t add_name(std::string_view str)
		{
			auto it = std::find_if(
				m_names.begin(), m_names.end(), [&str](const auto &el) { return el == str; });
			if (it == m_names.end()) {
				m_names.emplace_back(str);
				return m_names.size() - 1;
			}
			return std::distance(m_names.begin(), it);
		}

		size_t get_cell_index(std::string_view str)
		{
			auto it = std::find_if(
				m_cellvars.begin(), m_cellvars.end(), [&str](const auto &el) { return el == str; });
			if (it != m_cellvars.end()) { return std::distance(m_cellvars.begin(), it); }

			it = std::find_if(
				m_freevars.begin(), m_freevars.end(), [&str](const auto &el) { return el == str; });
			ASSERT(it != m_freevars.end());
			return std::distance(m_freevars.begin(), it) + m_cellvars.size();
		}

		void add_stack_size(size_t size) { m_stack_size += size; }
	};

	using FunctionsMap = std::unordered_map<std::string, FunctionInfo>;

	FunctionInfo m_module;
	FunctionsMap m_function_map;
	std::optional<std::reference_wrapper<typename FunctionsMap::value_type>> m_current_function;
	std::stack<decltype(LinearScanRegisterAllocation::value2mem_map)> m_register_mapping;
	std::stack<size_t> m_current_operation_index;
	std::stack<std::vector<Block *>> m_sorted_blocks;
	mlir::func::FuncOp m_parent_fn;

	bool m_writing_to_module{ true };

	FunctionInfo &current_function()
	{
		if (m_writing_to_module) { return m_module; }
		ASSERT(m_current_function.has_value());
		return m_current_function->get().second;
	}

	struct BlockLabel
	{
		mlir::Block *m_block;
		std::shared_ptr<Label> m_label;
	};

	struct BlockOffset
	{
		mlir::Block *m_block;
		// offset from function start
		size_t m_offset;
	};

	std::vector<BlockLabel> m_block_labels;
	std::vector<BlockOffset> m_block_offsets;

	PythonBytecodeEmitter() = default;

	template<typename InstructionT, typename... Args> void emit(Args &&...args)
	{
		auto instruction = std::make_unique<InstructionT>(std::forward<Args>(args)...);
		current_function().instructions.push_back(std::move(instruction));
	}

	size_t add_name(std::string_view str) { return current_function().add_name(str); }

	size_t get_cell_index(std::string_view str) { return current_function().get_cell_index(str); }

	size_t add_const(::py::Value value) { return current_function().add_const(std::move(value)); }

	size_t add_const(bool value) { return add_const(::py::NameConstant{ value }); }

	size_t add_const(::py::NoneType) { return add_const(::py::NameConstant{ ::py::NoneType{} }); }

	size_t add_const(llvm::StringRef str)
	{
		::py::String s{ str.str() };
		return add_const(std::move(s));
	}

	size_t add_const(llvm::APSInt int_value)
	{
		::py::BigIntType big_int_value{};
		if (int_value.isZero()) {
			big_int_value.get_mpz_t()->_mp_size = 0;
		} else {
			mpz_init2(big_int_value.get_mpz_t(), int_value.getBitWidth());
			big_int_value.get_mpz_t()->_mp_size =
				int_value.getNumWords() * (int_value.isNegative() ? -1 : 1);
			std::copy_n(
				int_value.getRawData(), int_value.getNumWords(), big_int_value.get_mpz_t()->_mp_d);
		}
		::py::Number value{ std::move(big_int_value) };
		return add_const(std::move(value));
	}

	size_t add_const(double value) { return add_const(::py::Number{ value }); }

	size_t add_const(std::vector<std::byte> bytes)
	{
		return add_const(::py::Bytes{ std::move(bytes) });
	}

	Register get_register(const mlir::Value &value) const
	{
		const auto mem = m_register_mapping.top().at(value);
		ASSERT(std::holds_alternative<LinearScanRegisterAllocation::Reg>(mem));
		const auto reg = std::get<LinearScanRegisterAllocation::Reg>(mem).idx;
		ASSERT(reg <= std::numeric_limits<Register>::max());

		return reg;
	}

	Register get_name_idx(StringRef name) const
	{
		// llvm::outs() << const_cast<mlir::func::FuncOp &>(m_parent_fn).getName() << '\n';
		auto names = const_cast<mlir::func::FuncOp &>(m_parent_fn).getOperation()->getAttr("names");
		ASSERT(names);
		auto names_array = names.cast<mlir::ArrayAttr>();
		auto it =
			std::find_if(names_array.begin(), names_array.end(), [&name](mlir::Attribute attr) {
				return attr.cast<mlir::StringAttr>().getValue() == name;
			});
		ASSERT(it != names_array.end());
		const auto idx = std::distance(names_array.begin(), it);
		ASSERT(idx <= std::numeric_limits<Register>::max());
		return idx;
	}

	Register get_local_idx(StringRef name) const
	{
		ASSERT(!m_writing_to_module);
		const auto &varnames = m_current_function->get().second.m_varnames;
		auto it = std::find_if(varnames.begin(), varnames.end(), [&name](const std::string &el) {
			return el == name;
		});
		ASSERT(it != varnames.end());
		const auto idx = std::distance(varnames.begin(), it);
		ASSERT(idx <= std::numeric_limits<Register>::max());
		return idx;
	}

	void enter_function_op(mlir::func::FuncOp op)
	{
		auto extract_str_array = [&op](std::string_view array_name,
									 std::vector<std::string> &func_array) {
			auto attr = op.getOperation()->getAttr(array_name);
			if (attr) {
				auto array = attr.cast<mlir::ArrayAttr>();
				func_array.reserve(array.size());
				std::transform(array.begin(),
					array.end(),
					std::back_inserter(func_array),
					[](mlir::Attribute attr) {
						return attr.cast<mlir::StringAttr>().getValue().str();
					});
			}
		};

		extract_str_array("locals", current_function().m_varnames);
		extract_str_array("names", current_function().m_names);
		extract_str_array("cellvars", current_function().m_cellvars);
		extract_str_array("freevars", current_function().m_freevars);
		current_function().m_cell2arg = std::vector<size_t>(
			current_function().m_cellvars.size(), op.getFunctionType().getNumInputs());
		if (op.getAllArgAttrs()
			&& std::any_of(op.getAllArgAttrs().begin(),
				op.getAllArgAttrs().end(),
				[](mlir::Attribute arg_attr) {
					auto vararg =
						arg_attr.cast<mlir::DictionaryAttr>().getAs<mlir::BoolAttr>("llvm.vararg");
					return vararg && vararg.getValue();
				})) {
			current_function().set_varargs();
		}

		if (op.getAllArgAttrs()
			&& std::any_of(op.getAllArgAttrs().begin(),
				op.getAllArgAttrs().end(),
				[](mlir::Attribute arg_attr) {
					auto vararg =
						arg_attr.cast<mlir::DictionaryAttr>().getAs<mlir::BoolAttr>("llvm.kwarg");
					return vararg && vararg.getValue();
				})) {
			current_function().set_kwargs();
		}

		if (op.getAllArgAttrs()) {
			auto kwonlyarg_count = std::count_if(op.getAllArgAttrs().begin(),
				op.getAllArgAttrs().end(),
				[](mlir::Attribute arg_attr) {
					auto vararg = arg_attr.cast<mlir::DictionaryAttr>().getAs<mlir::BoolAttr>(
						"llvm.kwonlyarg");
					return vararg && vararg.getValue();
				});
			current_function().set_kwonlyarg_count(kwonlyarg_count);
		}

		if (auto is_generator = op->getAttrOfType<mlir::BoolAttr>("is_generator");
			is_generator && is_generator.getValue()) {
			current_function().set_is_generator();
		}

		if (auto is_generator = op->getAttrOfType<mlir::BoolAttr>("is_class");
			is_generator && is_generator.getValue()) {
			current_function().set_is_class();
		}

		ASSERT(!m_writing_to_module || op.getFunctionType().getNumInputs() == 0);
		if (!m_writing_to_module) {
			current_function().m_arg_count =
				op.getFunctionType().getNumInputs() - current_function().m_kwonly_arg_count
				- current_function().m_flags.is_set(CodeFlags::Flag::VARARGS)
				- current_function().m_flags.is_set(CodeFlags::Flag::VARKEYWORDS);
			for (const auto &arg :
				llvm::enumerate(current_function().m_varnames
								| std::ranges::views::take(current_function().m_arg_count))) {
				const auto &arg_idx = arg.index();
				const auto &arg_name = arg.value();
				if (auto it = std::ranges::find(current_function().m_cellvars, arg_name);
					it != current_function().m_cellvars.end()) {
					const auto cell_idx = std::distance(current_function().m_cellvars.begin(), it);
					current_function().m_cell2arg[cell_idx] = arg_idx;
				}
			}

			if (current_function().m_flags.is_set(CodeFlags::Flag::VARARGS)) {
				auto arg_name = std::find_if(op.getAllArgAttrs().begin(),
					op.getAllArgAttrs().end(),
					[](mlir::Attribute arg_attr) {
						auto vararg = arg_attr.cast<mlir::DictionaryAttr>().getAs<mlir::BoolAttr>(
							"llvm.vararg");
						return vararg && vararg.getValue();
					})
									->cast<mlir::DictionaryAttr>()
									.getAs<mlir::StringAttr>("llvm.name")
									.getValue();
				if (auto it = std::ranges::find(current_function().m_cellvars, arg_name);
					it != current_function().m_cellvars.end()) {
					const auto cell_idx = std::distance(current_function().m_cellvars.begin(), it);
					current_function().m_cell2arg[cell_idx] = current_function().m_arg_count;
				}
			}

			if (current_function().m_flags.is_set(CodeFlags::Flag::VARKEYWORDS)) {
				auto arg_name = std::find_if(op.getAllArgAttrs().begin(),
					op.getAllArgAttrs().end(),
					[](mlir::Attribute arg_attr) {
						auto kwarg = arg_attr.cast<mlir::DictionaryAttr>().getAs<mlir::BoolAttr>(
							"llvm.kwarg");
						return kwarg && kwarg.getValue();
					})
									->cast<mlir::DictionaryAttr>()
									.getAs<mlir::StringAttr>("llvm.name")
									.getValue();
				if (auto it = std::ranges::find(current_function().m_cellvars, arg_name);
					it != current_function().m_cellvars.end()) {
					const auto cell_idx = std::distance(current_function().m_cellvars.begin(), it);
					current_function().m_cell2arg[cell_idx] =
						current_function().m_arg_count
						+ current_function().m_flags.is_set(CodeFlags::Flag::VARARGS);
				}
			}
		}
		m_parent_fn = op;
	}

	template<typename OpType> LogicalResult emitOperation(OpType &op);

	void push(Register value);
};

template<> LogicalResult PythonBytecodeEmitter::emitOperation(Operation &op)
{
	return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
		// Builtin ops.
		.Case<mlir::ModuleOp>([this](auto op) {
			m_current_operation_index.push(0);
			return emitOperation(op);
		})
		.Case<mlir::emitpybytecode::ConstantOp,
			mlir::emitpybytecode::StoreFastOp,
			mlir::emitpybytecode::StoreGlobalOp,
			mlir::emitpybytecode::StoreNameOp,
			mlir::emitpybytecode::StoreDerefOp,
			mlir::emitpybytecode::LoadFastOp,
			mlir::emitpybytecode::LoadNameOp,
			mlir::emitpybytecode::LoadGlobalOp,
			mlir::emitpybytecode::LoadClosureOp,
			mlir::emitpybytecode::LoadDerefOp,
			mlir::emitpybytecode::DeleteFastOp,
			mlir::emitpybytecode::DeleteNameOp,
			mlir::emitpybytecode::DeleteGlobalOp,
			mlir::emitpybytecode::UnpackSequenceOp,
			mlir::emitpybytecode::BuildSliceOp>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::BinaryOp,
			mlir::emitpybytecode::UnaryOp,
			mlir::emitpybytecode::InplaceOp>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::FunctionCallOp,
			mlir::emitpybytecode::FunctionCallExOp,
			mlir::emitpybytecode::FunctionCallWithKeywordsOp>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::func::FuncOp>([this](auto op) {
			auto prev = m_parent_fn;
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			m_parent_fn = prev;
			return success();
		})
		.Case<mlir::emitpybytecode::MakeFunction>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::func::ReturnOp>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::Yield,
			mlir::emitpybytecode::YieldFrom,
			mlir::emitpybytecode::YieldFromIter>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::cf::BranchOp,
			mlir::emitpybytecode::JumpIfFalse,
			mlir::emitpybytecode::JumpIfNotException>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::Compare>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::LoadAssertionError,
			mlir::emitpybytecode::RaiseVarargs,
			mlir::emitpybytecode::ReRaiseOp>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::BuildDict,
			mlir::emitpybytecode::DictUpdate,
			mlir::emitpybytecode::DictAdd,
			mlir::emitpybytecode::BuildTuple,
			mlir::emitpybytecode::BuildSet,
			mlir::emitpybytecode::SetAdd,
			mlir::emitpybytecode::SetUpdate,
			mlir::emitpybytecode::BuildList,
			mlir::emitpybytecode::ListExtend,
			mlir::emitpybytecode::ListAppend,
			mlir::emitpybytecode::ListToTuple>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::LoadAttribute, mlir::emitpybytecode::LoadMethod>(
			[this](auto op) {
				if (emitOperation(op).failed()) { return failure(); };
				m_current_operation_index.top()++;
				return success();
			})
		.Case<mlir::emitpybytecode::BinarySubscript,
			mlir::emitpybytecode::StoreSubscript,
			mlir::emitpybytecode::DeleteSubscript>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::StoreAttribute>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Case<mlir::emitpybytecode::Push, mlir::emitpybytecode::Pop, mlir::emitpybytecode::Move>(
			[this](auto op) {
				if (emitOperation(op).failed()) { return failure(); };
				return success();
			})
		.Case<mlir::emitpybytecode::SetupExceptionHandle,
			mlir::emitpybytecode::SetupWith,
			mlir::emitpybytecode::WithExceptStart,
			mlir::emitpybytecode::ClearExceptionState,
			mlir::emitpybytecode::LeaveExceptionHandle>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Case<mlir::emitpybytecode::ImportName,
			mlir::emitpybytecode::ImportFrom,
			mlir::emitpybytecode::ImportAll>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Case<mlir::emitpybytecode::CastToBool>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Case<mlir::emitpybytecode::ForIter, mlir::emitpybytecode::GetIter>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Case<mlir::emitpybytecode::LoadBuildClass>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Default([this](Operation *op) {
			llvm::outs() << "Operation " << op->getName() << " not implemented\n";
			return failure();
		});
}

void PythonBytecodeEmitter::push(Register value)
{
	if (m_current_function.has_value()) { m_current_function->get().second.add_stack_size(1); }
	emit<Push>(value);
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadBuildClass &op)
{
	emit<LoadBuildClass>(get_register(op.getClassBuilder()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ImportName &op)
{
	emit<ImportName>(get_register(op.getModule()),
		add_name(op.getName()),
		get_register(op.getFromList()),
		get_register(op.getLevel()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ImportFrom &op)
{
	emit<ImportFrom>(
		get_register(op.getObject()), add_name(op.getName()), get_register(op.getModule()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ImportAll &op)
{
	emit<ImportStar>(get_register(op.getModule()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::CastToBool &op)
{
	emit<ToBool>(get_register(op.getOutput()), get_register(op.getValue()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ForIter &op)
{
	// auto this_block = std::find(
	// 	m_sorted_blocks.top().begin(), m_sorted_blocks.top().end(), op.getOperation()->getBlock());
	// ASSERT(*(this_block + 1) == op.body());

	auto exit_label =
		m_block_labels.emplace_back(op.getContinuation(), std::make_shared<Label>("", 0)).m_label;
	auto body_label =
		m_block_labels.emplace_back(op.getBody(), std::make_shared<Label>("", 0)).m_label;

	emit<ForIter>(get_register(op.getValue()),
		get_register(op.getIterator()),
		std::move(body_label),
		std::move(exit_label));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::GetIter &op)
{
	emit<GetIter>(get_register(op.getIterator()), get_register(op.getIterable()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::SetupExceptionHandle &op)
{
	auto &handler_label =
		m_block_labels.emplace_back(op.getHandler(), std::make_shared<Label>("", 0));
	emit<SetupExceptionHandling>(handler_label.m_label);
	auto &body_label = m_block_labels.emplace_back(op.getBody(), std::make_shared<Label>("", 0));
	emit<Jump>(body_label.m_label);
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::SetupWith &op)
{
	auto &handler_label =
		m_block_labels.emplace_back(op.getHandler(), std::make_shared<Label>("", 0));
	emit<SetupWith>(handler_label.m_label);
	auto &body_label = m_block_labels.emplace_back(op.getBody(), std::make_shared<Label>("", 0));
	emit<Jump>(body_label.m_label);
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::WithExceptStart &op)
{
	emit<WithExceptStart>(get_register(op.getOutput()), get_register(op.getExitMethod()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LeaveExceptionHandle &)
{
	emit<LeaveExceptionHandling>();
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ClearExceptionState &)
{
	emit<ClearExceptionState>();
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::func::FuncOp &op)
{
	LinearScanRegisterAllocation register_allocation{};
	register_allocation.analyse(op, mlir::OpBuilder{ op.getContext() });
	m_register_mapping.push(register_allocation.value2mem_map);

	std::string prev_func;
	const bool is_module_entry = op.isPrivate() && op.getSymName() == "__hidden_init__";
	if (!is_module_entry) {
		auto current_func = m_function_map.emplace(op.getSymName(), FunctionInfo{});
		prev_func = m_current_function.has_value() ? m_current_function->get().first : "";
		m_current_function = *current_func.first;
	}
	FunctionInfo &function_info = is_module_entry ? m_module : m_current_function->get().second;

	const bool prev_writing_to_module = std::exchange(m_writing_to_module, is_module_entry);

	m_current_operation_index.push(0);

	auto &region = op.getRegion();

	enter_function_op(op);

	m_sorted_blocks.push(sortBlocks(region));
	// llvm::outs() << "-----------------------------------------------\n";
	// for (auto *block : m_sorted_blocks.top()) {
	// 	block->print(llvm::outs());
	// 	llvm::outs() << '\n';
	// }
	for (size_t op_idx = 0; auto *block : m_sorted_blocks.top()) {
		m_block_offsets.emplace_back(block, op_idx);
		if (!sortTopologically(block)) { std::abort(); }
		const auto start = function_info.instructions.size();
		for (auto &op : block->getOperations()) {
			if (failed(emitOperation(op))) { return failure(); }
		}
		op_idx += function_info.instructions.size() - start;
	}

	if (!prev_writing_to_module) {
		m_current_function = *m_function_map.find(prev_func);
	} else {
		m_current_function = {};
	}
	m_register_mapping.pop();
	m_sorted_blocks.pop();
	m_writing_to_module = prev_writing_to_module;

	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::MakeFunction &op)
{
	for (const auto &default_ : op.getDefaults()) { push(get_register(default_)); }
	for (const auto &default_ : op.getKwDefaults()) { push(get_register(default_)); }

	const size_t defaults_size = op.getDefaults().size();
	const size_t kw_defaults_size = op.getKwDefaults().size();
	auto captures = op.getCaptures() ? get_register(op.getCaptures()) : std::optional<Register>{};
	emit<MakeFunction>(
		get_register(op), get_register(op.getSymName()), defaults_size, kw_defaults_size, captures);

	for (size_t i = 0; i < defaults_size + kw_defaults_size; ++i) { emit<Pop>(); }

	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::cf::BranchOp &op)
{
	auto *bb = op.getDest();
	auto &label = m_block_labels.emplace_back(bb, std::make_shared<Label>("", 0));
	emit<Jump>(label.m_label);
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadAssertionError &op)
{
	emit<LoadAssertionError>(get_register(op.getOutput()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DictUpdate &op)
{
	emit<DictUpdate>(get_register(op.getDict()), get_register(op.getMappable()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DictAdd &op)
{
	emit<DictAdd>(
		get_register(op.getDict()), get_register(op.getKey()), get_register(op.getValue()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BuildDict &op)
{
	ASSERT(op.getKeys().size() == op.getValues().size());
	for (const auto &key : op.getKeys()) { push(get_register(key)); }
	for (const auto &value : op.getValues()) { push(get_register(value)); }
	emit<BuildDict>(get_register(op.getOutput()), op.getKeys().size());
	for (size_t i = 0; i < op.getKeys().size() + op.getValues().size(); ++i) { emit<Pop>(); }
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BuildList &op)
{
	for (const auto &el : op.getElements()) { push(get_register(el)); }
	emit<BuildList>(get_register(op.getOutput()), op.getElements().size());
	for (size_t i = 0; i < op.getElements().size(); ++i) { emit<Pop>(); }
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ListExtend &op)
{
	emit<ListExtend>(get_register(op.getList()), get_register(op.getIterable()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ListAppend &op)
{
	emit<ListAppend>(get_register(op.getList()), get_register(op.getValue()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ListToTuple &op)
{
	emit<ListToTuple>(get_register(op.getTuple()), get_register(op.getList()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BuildTuple &op)
{
	for (const auto &el : op.getElements()) { push(get_register(el)); }
	emit<BuildTuple>(get_register(op.getOutput()), op.getElements().size());
	for (size_t i = 0; i < op.getElements().size(); ++i) { emit<Pop>(); }
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BuildSet &op)
{
	for (const auto &el : op.getElements()) { push(get_register(el)); }
	emit<BuildSet>(get_register(op.getOutput()), op.getElements().size());
	for (size_t i = 0; i < op.getElements().size(); ++i) { emit<Pop>(); }
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::SetAdd &op)
{
	emit<SetAdd>(get_register(op.getSet()), get_register(op.getElement()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::SetUpdate &op)
{
	emit<SetUpdate>(get_register(op.getSet()), get_register(op.getIterable()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadAttribute &op)
{
	emit<LoadAttr>(
		get_register(op.getOutput()), get_register(op.getSelf()), add_name(op.getAttr()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadMethod &op)
{
	emit<LoadMethod>(
		get_register(op.getMethod()), get_register(op.getSelf()), add_name(op.getMethodName()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BinarySubscript &op)
{
	emit<BinarySubscript>(
		get_register(op.getOutput()), get_register(op.getSelf()), get_register(op.getSubscript()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreSubscript &op)
{
	emit<StoreSubscript>(
		get_register(op.getSelf()), get_register(op.getSubscript()), get_register(op.getValue()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DeleteSubscript &op)
{
	emit<DeleteSubscript>(get_register(op.getSelf()), get_register(op.getSubscript()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreAttribute &op)
{
	emit<StoreAttr>(
		get_register(op.getSelf()), get_register(op.getValue()), get_name_idx(op.getAttribute()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::RaiseVarargs &op)
{
	if (auto cause = op.getCause()) {
		emit<RaiseVarargs>(get_register(op.getException()), get_register(cause));
	} else {
		emit<RaiseVarargs>(get_register(op.getException()));
	}
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ReRaiseOp &op)
{
	emit<ReRaise>();
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::JumpIfFalse &op)
{
	auto this_block = std::find(
		m_sorted_blocks.top().begin(), m_sorted_blocks.top().end(), op.getOperation()->getBlock());
	ASSERT(*(this_block + 1) == op.getFalseDest() || *(this_block + 1) == op.getTrueDest());

	if (*(this_block + 1) == op.getTrueDest()) {
		auto *bb = op.getFalseDest();
		auto &label = m_block_labels.emplace_back(bb, std::make_shared<Label>("", 0));
		emit<JumpIfFalse>(get_register(op.getCond()), label.m_label);
	} else {
		auto *bb = op.getTrueDest();
		auto &label = m_block_labels.emplace_back(bb, std::make_shared<Label>("", 0));
		emit<JumpIfTrue>(get_register(op.getCond()), label.m_label);
	}
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::JumpIfNotException &op)
{
	auto this_block = std::find(
		m_sorted_blocks.top().begin(), m_sorted_blocks.top().end(), op.getOperation()->getBlock());
	ASSERT(*(this_block + 1) == op.getFalseDest() || *(this_block + 1) == op.getTrueDest());

	if (*(this_block + 1) == op.getTrueDest()) {
		auto *bb = op.getFalseDest();
		auto &label = m_block_labels.emplace_back(bb, std::make_shared<Label>("", 0));
		emit<JumpIfExceptionMatch>(get_register(op.getObjectType()), label.m_label);
	} else {
		auto *bb = op.getTrueDest();
		auto &label = m_block_labels.emplace_back(bb, std::make_shared<Label>("", 0));
		emit<JumpIfNotExceptionMatch>(get_register(op.getObjectType()), label.m_label);
	}

	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ConstantOp &op)
{
	return llvm::TypeSwitch<mlir::Attribute, LogicalResult>(op.getValue())
		.Case<FloatAttr>([this, &op](FloatAttr f) {
			const auto value_as_double = f.getValueAsDouble();
			const auto idx = add_const(value_as_double);
			emit<LoadConst>(get_register(op.getResult()), idx);
			return success();
		})
		.Case<BoolAttr>([this, &op](BoolAttr b) {
			const auto value_as_bool = b.getValue();
			const auto idx = add_const(value_as_bool);
			emit<LoadConst>(get_register(op.getResult()), idx);
			return success();
		})
		.Case<UnitAttr>([this, &op](UnitAttr b) {
			const auto idx = add_const(::py::NoneType{});
			emit<LoadConst>(get_register(op.getResult()), idx);
			return success();
		})
		.Case<StringAttr>([this, &op](StringAttr str) {
			const auto idx = add_const(str.getValue());
			emit<LoadConst>(get_register(op.getResult()), idx);
			return success();
		})
		.Case<IntegerAttr>([this, &op](IntegerAttr int_attr) {
			const auto idx = add_const(int_attr.getAPSInt());
			emit<LoadConst>(get_register(op.getResult()), idx);
			return success();
		})
		.Case<DenseIntElementsAttr>([this, &op](DenseIntElementsAttr bytes_attr) {
			std::vector<std::byte> bytes;
			for (const auto &el : bytes_attr) {
				ASSERT(el.isIntN(8));
				bytes.push_back(std::byte{ *bit_cast<const uint8_t *>(el.getRawData()) });
			}
			const auto idx = add_const(std::move(bytes));
			emit<LoadConst>(get_register(op.getResult()), idx);
			return success();
		});
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreFastOp &op)
{
	emit<StoreFast>(get_local_idx(op.getName()), get_register(op.getOperand()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreGlobalOp &op)
{
	emit<StoreGlobal>(get_name_idx(op.getName()), get_register(op.getOperand()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreNameOp &op)
{
	emit<StoreName>(op.getName().str(), get_register(op.getOperand()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreDerefOp &op)
{
	emit<StoreDeref>(get_cell_index(op.getName().str()), get_register(op.getOperand()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::FunctionCallOp &op)
{
	const auto arg_size = op.getArgs().size();
	for (const auto &arg : op.getArgs()) { push(get_register(arg)); }
	emit<FunctionCall>(get_register(op.getCallee()), arg_size, 0);
	for (size_t i = 0; i < arg_size; ++i) { emit<Pop>(); }
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(
	mlir::emitpybytecode::FunctionCallWithKeywordsOp &op)
{
	std::vector<Register> arg_registers;
	std::vector<Register> kwarg_registers;
	std::vector<Register> keywords_registers;

	arg_registers.reserve(op.getArgs().size());
	for (const auto &arg : op.getArgs()) { arg_registers.push_back(get_register(arg)); }

	kwarg_registers.reserve(op.getKwargs().size());
	keywords_registers.reserve(op.getKeywords().size());
	for (auto [keyword, value] :
		llvm::zip(op.getKeywords().getValues<mlir::StringRef>(), op.getKwargs())) {
		kwarg_registers.push_back(get_register(value));
		keywords_registers.push_back(add_name(keyword));
	}

	emit<FunctionCallWithKeywords>(get_register(op.getCallee()),
		std::move(arg_registers),
		std::move(kwarg_registers),
		std::move(keywords_registers));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::FunctionCallExOp &op)
{
	emit<FunctionCallEx>(get_register(op.getCallee()),
		op.getArgs() ? get_register(op.getArgs()) : Register{ 0 },
		op.getKwargs() ? get_register(op.getKwargs()) : Register{ 0 },
		op.getArgs() != nullptr,
		op.getKwargs() != nullptr);
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::func::ReturnOp &op)
{
	ASSERT(op.getOperands().size() == 1);
	emit<ReturnValue>(get_register(op.getOperands().back()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::Yield &op)
{
	emit<YieldValue>(get_register(op.getValue()));
	// TODO: optimise away the YieldLoad when the received value is never used
	//       would have to happen during op lowering to EmitPythonBytecode dialect in order to avoid
	//       register allocation of `received`
	emit<YieldLoad>(get_register(op.getReceived()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::YieldFrom &op)
{
	emit<YieldFrom>(get_register(op.getReceived()),
		get_register(op.getIterator()),
		get_register(op.getValue()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::YieldFromIter &op)
{
	emit<GetYieldFromIter>(get_register(op.getIterator()), get_register(op.getIterable()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::Push &op)
{
	const auto src = op.getSrc();
	ASSERT(src <= std::numeric_limits<Register>::max());
	push(src);
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::Pop &op)
{
	const auto dst = op.getDst();
	ASSERT(dst <= std::numeric_limits<Register>::max());
	emit<Pop>(dst);
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::Move &op)
{
	const auto src = op.getSrc();
	ASSERT(src <= std::numeric_limits<Register>::max());

	const auto dst = op.getDst();
	ASSERT(dst <= std::numeric_limits<Register>::max());

	emit<Move>(dst, src);

	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BinaryOp &op)
{
	emit<BinaryOperation>(get_register(op.getOutput()),
		get_register(op.getLhs()),
		get_register(op.getRhs()),
		static_cast<BinaryOperation::Operation>(op.getOperationType()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::UnaryOp &op)
{
	emit<Unary>(get_register(op.getOutput()),
		get_register(op.getInput()),
		static_cast<Unary::Operation>(op.getOperationType()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::InplaceOp &op)
{
	emit<InplaceOp>(get_register(op.getDst()),
		get_register(op.getSrc()),
		static_cast<InplaceOp::Operation>(op.getOperationType()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadNameOp &op)
{
	emit<LoadName>(get_register(op.getOutput()), op.getName().str());
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadFastOp &op)
{
	emit<LoadFast>(get_register(op.getOutput()), get_local_idx(op.getName()), op.getName().str());
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadGlobalOp &op)
{
	emit<LoadGlobal>(get_register(op.getOutput()), add_name(op.getName()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadDerefOp &op)
{
	emit<LoadDeref>(get_register(op.getOutput()), get_cell_index(op.getName()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadClosureOp &op)
{
	emit<LoadClosure>(get_register(op.getOutput()), get_cell_index(op.getName()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DeleteNameOp &op)
{
	emit<DeleteName>(add_const(op.getName()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DeleteFastOp &op)
{
	emit<DeleteFast>(get_local_idx(op.getName()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DeleteGlobalOp &op)
{
	emit<DeleteGlobal>(add_const(op.getName()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::UnpackSequenceOp &op)
{
	std::vector<Register> unpacked_values;
	for (const auto &el : op.getUnpackedValues()) { unpacked_values.push_back(get_register(el)); }
	emit<UnpackSequence>(unpacked_values, get_register(op.getIterable()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BuildSliceOp &op)
{
	emit<BuildSlice>(get_register(op.getOutput()),
		get_register(op.getLower()),
		get_register(op.getUpper()),
		get_register(op.getStep()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::Compare &op)
{
	emit<CompareOperation>(get_register(op.getOutput()),
		get_register(op.getLhs()),
		get_register(op.getRhs()),
		static_cast<CompareOperation::Comparisson>(op.getPredicate()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::ModuleOp &module_)
{
	auto &module_region = module_.getBodyRegion();
	ASSERT(module_region.getBlocks().size() == 1);
	auto fn = std::find_if(module_region.getBlocks().back().getOperations().begin(),
		module_region.getBlocks().back().getOperations().end(),
		[](mlir::Operation &op) {
			auto fn = mlir::cast<mlir::func::FuncOp>(op);
			if (fn) { return fn.isPrivate() && fn.getSymName() == "__hidden_init__"; }
			return false;
		});
	ASSERT(fn != module_region.getBlocks().back().getOperations().end());
	m_parent_fn = mlir::cast<mlir::func::FuncOp>(*fn);

	if (failed(emitOperation(m_parent_fn))) { return failure(); }

	for (auto &op : module_region.getBlocks().back().getOperations()) {
		ASSERT(mlir::isa<mlir::func::FuncOp>(op));
		if (mlir::cast<mlir::func::FuncOp>(op).isPrivate()
			&& mlir::cast<mlir::func::FuncOp>(op).getSymName() == "__hidden_init__") {
			continue;
		}
		if (failed(emitOperation(op))) { return failure(); }
	}

	for (auto &block_label : m_block_labels) {
		auto *bb = block_label.m_block;
		auto &label = block_label.m_label;
		auto it = std::find_if(m_block_offsets.begin(),
			m_block_offsets.end(),
			[bb](const auto &el) { return el.m_block == bb; });
		ASSERT(it != m_block_offsets.end());
		label->set_position(it->m_offset);
	}

	size_t instruction_idx{ 0 };
	for (const auto &ins : m_module.instructions) { ins->relocate(instruction_idx++); }
	for (const auto &[_, func] : m_function_map) {
		instruction_idx = 0;
		for (const auto &ins : func.instructions) { ins->relocate(instruction_idx++); }
	}

	return success();
}


std::shared_ptr<BytecodeProgram> translateToPythonBytecode(Operation *op)
{
	if (mlir::verify(op, /*verifyRecursively*/ true).failed()) {
		std::cerr << "Invalid Python bytecode IR\n";
		return nullptr;
	}

	DialectRegistry registry;
	registry.insert<emitpybytecode::EmitPythonBytecodeDialect>();
	PythonBytecodeEmitter emitter;
	if (failed(emitter.emitOperation(*op))) { return nullptr; }

	FunctionBlocks func_blocks;
	InstructionVector instructions = std::move(emitter.m_module.instructions);
	{
		FunctionBlock fb_module{
			.metadata =
				FunctionMetaData{
					.function_name = "__main__",
					.register_count = 32,
					.stack_size = 2,
					.names = std::move(emitter.m_module.m_names),
					.consts = std::move(emitter.m_module.m_consts),
				},
			.blocks = std::move(instructions),
		};
		func_blocks.functions.push_back(std::move(fb_module));
	}
	for (auto &&[name, fn] : emitter.m_function_map) {
		const auto stack_size = fn.m_varnames.size() + fn.m_stack_size;
		FunctionBlock fb{
			.metadata =
				FunctionMetaData{
					.function_name = name,
					.register_count = 32,
					.stack_size = stack_size,
					.cellvars = std::move(fn.m_cellvars),
					.varnames = std::move(fn.m_varnames),
					.freevars = std::move(fn.m_freevars),
					.names = std::move(fn.m_names),
					.arg_count = fn.m_arg_count,
					.kwonly_arg_count = fn.m_kwonly_arg_count,
					.cell2arg = std::move(fn.m_cell2arg),
					.consts = std::move(fn.m_consts),
					.flags = std::move(fn.m_flags),
				},
			.blocks = std::move(fn.instructions),
		};
		func_blocks.functions.push_back(std::move(fb));
	}
	return BytecodeProgram::create(std::move(func_blocks), "", { "" });
}

}// namespace codegen