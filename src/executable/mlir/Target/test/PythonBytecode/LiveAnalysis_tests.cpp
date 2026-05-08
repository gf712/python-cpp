#include "Target/PythonBytecode/LiveAnalysis.hpp"
#include "Target/PythonBytecode/RegisterAllocationLogger.hpp"

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/Dialect.hpp"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "utilities.hpp"

#include "gtest/gtest.h"
#include <mlir/IR/Block.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <spdlog/spdlog.h>

using namespace codegen;


namespace {

constexpr const char *kLiveAnalysis = R"(
module attributes {llvm.argv = ["live_analysis.py"]} {
  func.func private @__hidden_init__() -> !python.object attributes {names = ["split"]} {
    %1 = "emitpybytecode.LOAD_NAME"() <{name = "test_fn"}> : () -> !python.object
    %2 = "emitpybytecode.LOAD_CONST"() <{value = "1"}> : () -> !python.object
    %3 = "emitpybytecode.CALL"(%1, %2) : (!python.object, !python.object) -> !python.object
    %4 = "emitpybytecode.TO_BOOL"(%3) : (!python.object) -> !python.object
    emitpybytecode.JUMP_IF_FALSE %4, ^bb1, ^bb4(%3 : !python.object)
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %5 = "emitpybytecode.LOAD_NAME"() <{name = "test_fn"}> : () -> !python.object
    %6 = "emitpybytecode.LOAD_CONST"() <{value = "2"}> : () -> !python.object
    %7 = "emitpybytecode.CALL"(%5, %6) : (!python.object, !python.object) -> !python.object
    %8 = "emitpybytecode.UNARY"(%7) <{operation_type = 3 : ui8}> : (!python.object) -> !python.object
    %9 = "emitpybytecode.TO_BOOL"(%8) : (!python.object) -> !python.object
    emitpybytecode.JUMP_IF_FALSE %9, ^bb3, ^bb4(%8 : !python.object)
  ^bb3:  // pred: ^bb2
    %10 = "emitpybytecode.LOAD_NAME"() <{name = "test_fn"}> : () -> !python.object
    %11 = "emitpybytecode.LOAD_CONST"() <{value = "3"}> : () -> !python.object
    %12 = "emitpybytecode.CALL"(%10, %11) : (!python.object, !python.object) -> !python.object
    cf.br ^bb4(%12 : !python.object)
  ^bb4(%18: !python.object):  // 3 preds: ^bb0, ^bb2, ^bb3
    %13 = "emitpybytecode.TO_BOOL"(%18) : (!python.object) -> !python.object
    emitpybytecode.JUMP_IF_FALSE %13, ^bb5, ^bb6
  ^bb5:
    %14 = "emitpybytecode.LOAD_NAME"() <{name = "print"}> : () -> !python.object
    %15 = "emitpybytecode.LOAD_NAME"() <{name = "bar"}> : () -> !python.object
    %16 = "emitpybytecode.CALL"(%14, %15) : (!python.object, !python.object) -> !python.object
    cf.br ^bb6
  ^bb6:
    %17 = "emitpybytecode.LOAD_CONST"() <{value}> : () -> !python.object
    return %17 : !python.object
  }
}
)";


class LiveAnalysisTest : public ::testing::Test
{
	mlir::MLIRContext m_context;

  protected:
	void SetUp() override
	{
		// Load required dialects
		m_context.getOrLoadDialect<mlir::func::FuncDialect>();
		m_context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
		m_context.getOrLoadDialect<mlir::emitpybytecode::EmitPythonBytecodeDialect>();
		m_context.getOrLoadDialect<mlir::py::PythonDialect>();

		// Disable debug logging for tests unless debugging
		auto logger = get_regalloc_logger();
		logger->set_level(spdlog::level::debug);
	}

	mlir::OwningOpRef<mlir::ModuleOp> parseLiveAnalysisIR()
	{
		return mlir::parseSourceString<mlir::ModuleOp>(kLiveAnalysis, &m_context);
	}
};
}// namespace

TEST_F(LiveAnalysisTest, LiveAnalysis)
{
	auto module = parseLiveAnalysisIR();
	ASSERT_TRUE(module) << "Failed to parse MLIR IR";

	auto funcs = module->getOps<mlir::func::FuncOp>();
	ASSERT_FALSE(funcs.empty()) << "No functions found in module";
	auto func = *funcs.begin();

	LiveAnalysis live_analysis;
	live_analysis.analyse(func);

	// Test 1: Verify alive_at_timestep is populated
	EXPECT_GT(live_analysis.alive_at_timestep.size(), 0) << "alive_at_timestep should not be empty";

	// Test 2: Collect all CALL results, UNARY results, and block arguments
	std::vector<mlir::Value> call_results;
	std::vector<mlir::Value> unary_results;
	mlir::BlockArgument block_arg_18;

	func.walk([&](mlir::Block *block) {
		// Find bb4 (the block with 3 predecessors and block argument)
		if (block->getNumArguments() == 1) { block_arg_18 = block->getArgument(0); }

		// Collect all CALL and UNARY results
		for (auto &op : block->getOperations()) {
			if (auto call_op = llvm::dyn_cast<mlir::emitpybytecode::FunctionCallOp>(op)) {
				call_results.push_back(call_op.getOutput());
			} else if (auto unary_op = llvm::dyn_cast<mlir::emitpybytecode::UnaryOp>(op)) {
				unary_results.push_back(unary_op.getResult());
			}
		}
	});

	ASSERT_TRUE(block_arg_18) << "Block argument %18 not found";
	EXPECT_EQ(call_results.size(), 4) << "Expected 4 CALL operations";
	EXPECT_EQ(unary_results.size(), 1) << "Expected 1 UNARY operation";

	// Test 3: Verify block_input_mappings contains mappings for values that flow into block_arg_18
	// According to the IR, %3, %8, and %12 should all map to block_arg_18
	// We'll check that at least one of the CALL/UNARY results maps to the block argument
	int mapped_count = 0;
	for (const auto &call_result : call_results) {
		auto it = live_analysis.block_input_mappings.find(call_result);
		if (it != live_analysis.block_input_mappings.end() && it->second.contains(block_arg_18)) {
			mapped_count++;
		}
	}
	for (const auto &unary_result : unary_results) {
		auto it = live_analysis.block_input_mappings.find(unary_result);
		if (it != live_analysis.block_input_mappings.end() && it->second.contains(block_arg_18)) {
			mapped_count++;
		}
	}

	EXPECT_GE(mapped_count, 3)
		<< "Expected at least 3 values to map to block_arg_18 (from 3 predecessors)";

	// Test 4: Verify that block_arg_18 appears in alive_at_timestep
	bool found_block_arg = false;
	bool found_block_arg_inputs = false;
	for (const auto &timestep : live_analysis.alive_at_timestep) {
		for (const auto &val : timestep) {
			if (std::holds_alternative<mlir::Value>(val)) {
				auto mlir_val = std::get<mlir::Value>(val);
				if (mlir_val == block_arg_18) { found_block_arg = true; }
			}
		}
	}

	EXPECT_TRUE(found_block_arg) << "Block argument %18 should appear in alive_at_timestep";

	// Test 5: Verify that CALL results are tracked in alive_at_timestep
	int found_call_results = 0;
	for (const auto &timestep : live_analysis.alive_at_timestep) {
		for (const auto &val : timestep) {
			if (std::holds_alternative<mlir::Value>(val)) {
				auto mlir_val = std::get<mlir::Value>(val);
				for (const auto &call_result : call_results) {
					if (mlir_val == call_result) {
						found_call_results++;
						break;
					}
				}
			}
		}
	}

	EXPECT_GT(found_call_results, 0)
		<< "CALL results should appear in alive_at_timestep (needed for register allocation)";
}

TEST_F(LiveAnalysisTest, BlockInfoVerification)
{
	auto module = parseLiveAnalysisIR();
	ASSERT_TRUE(module) << "Failed to parse MLIR IR";

	auto funcs = module->getOps<mlir::func::FuncOp>();
	ASSERT_FALSE(funcs.empty()) << "No functions found in module";
	auto func = *funcs.begin();

	LiveAnalysis live_analysis;
	live_analysis.analyse(func);

	// Collect blocks and their values for verification
	std::vector<mlir::Block *> blocks;
	std::map<mlir::Block *, std::vector<mlir::Value>> block_defs;
	std::map<mlir::Block *, std::vector<mlir::Value>> block_uses;

	func.walk([&](mlir::Block *block) {
		blocks.push_back(block);

		std::vector<mlir::Value> defs;
		std::vector<mlir::Value> uses;

		// Compute use/def for this block
		for (auto &op : block->getOperations()) {
			// Add operands to uses (if not already defined in this block)
			for (auto operand : op.getOperands()) {
				auto it = std::find(defs.begin(), defs.end(), operand);
				if (it == defs.end()) {
					auto it2 = std::find(uses.begin(), uses.end(), operand);
					if (it2 == uses.end()) { uses.push_back(operand); }
				}
			}

			// Add results to defs
			for (auto result : op.getResults()) {
				auto it = std::find(defs.begin(), defs.end(), result);
				if (it == defs.end()) { defs.push_back(result); }
			}
		}

		block_defs[block] = defs;
		block_uses[block] = uses;
	});

	ASSERT_GE(blocks.size(), 6) << "Expected at least 6 blocks (bb0-bb6)";

	// Test bb0 (entry block)
	auto *bb0 = blocks[0];
	EXPECT_EQ(block_uses[bb0].size(), 0) << "bb0 should have no uses (entry block)";
	EXPECT_EQ(block_defs[bb0].size(), 4) << "bb0 should define 4 values (%1, %2, %3, %4)";

	// Test bb2 (contains CALL and UNARY)
	// bb2 is the third block (index 2)
	auto *bb2 = blocks[2];
	EXPECT_EQ(block_uses[bb2].size(), 0) << "bb2 should have no uses (all values defined locally)";
	EXPECT_EQ(block_defs[bb2].size(), 5) << "bb2 should define 5 values (%5, %6, %7, %8, %9)";

	// Test bb4 (block with argument %18)
	// Find bb4 by looking for block with 1 argument
	auto *bb4 = blocks[4];

	ASSERT_NE(bb4, nullptr) << "bb4 with block argument should exist";

	// bb4 should use its block argument
	EXPECT_GE(block_uses[bb4].size(), 1) << "bb4 should use its block argument %18";

	// Verify that the block argument is used
	auto block_arg = bb4->getArgument(0);
	bool arg_in_uses = false;
	for (const auto &use : block_uses[bb4]) {
		if (use == block_arg) {
			arg_in_uses = true;
			break;
		}
	}
	EXPECT_TRUE(arg_in_uses) << "Block argument %18 should be in bb4's use set";

	// bb4 should define %13 (TO_BOOL result)
	EXPECT_EQ(block_defs[bb4].size(), 1) << "bb4 should define 1 value (%13)";

	// Test block_input_mappings for values flowing into bb4
	// bb4 receives values from 3 predecessors (bb0, bb2, bb3)
	// Each predecessor passes a value to bb4's block argument

	// Count how many predecessors pass values to bb4
	int pred_count = 0;
	std::vector<mlir::Value> values_to_bb4;

	for (auto *pred : bb4->getPredecessors()) {
		auto *terminator = pred->getTerminator();

		// Check if this is a conditional branch that passes values
		if (auto jump_if_false = llvm::dyn_cast<mlir::emitpybytecode::JumpIfFalse>(terminator)) {
			// Check true dest
			if (jump_if_false.getTrueDest() == bb4) {
				pred_count++;
				for (auto operand : jump_if_false.getTrueDestOperands()) {
					values_to_bb4.push_back(operand);
				}
			}
			// Check false dest
			if (jump_if_false.getFalseDest() == bb4) {
				pred_count++;
				for (auto operand : jump_if_false.getFalseDestOperands()) {
					values_to_bb4.push_back(operand);
				}
			}
		}
		// Check if this is an unconditional branch
		else if (auto branch = llvm::dyn_cast<mlir::cf::BranchOp>(terminator)) {
			if (branch.getDest() == bb4) {
				pred_count++;
				for (auto operand : branch.getDestOperands()) { values_to_bb4.push_back(operand); }
			}
		}
	}

	EXPECT_EQ(pred_count, 3) << "bb4 should have 3 predecessors passing values";
	EXPECT_EQ(values_to_bb4.size(), 3) << "bb4 should receive 3 values from its predecessors";

	// Verify that each value passed to bb4 is mapped to its block argument in block_input_mappings
	for (const auto &value : values_to_bb4) {
		auto it = live_analysis.block_input_mappings.find(value);
		EXPECT_NE(it, live_analysis.block_input_mappings.end())
			<< "Value passed to bb4 should be in block_input_mappings";
		if (it != live_analysis.block_input_mappings.end()) {
			EXPECT_TRUE(it->second.contains(block_arg))
				<< "Value should map to bb4's block argument";
		}
	}

	// Verify that the values are distinct and are coming from the expected blocks
	std::set<mlir::detail::ValueImpl *> unique_values;
	std::set<mlir::Block *> expected_incoming_blocks{
		blocks[0],
		blocks[2],
		blocks[3],
	};
	for (auto val : values_to_bb4) {
		auto *bb = val.getParentBlock();
		ASSERT(expected_incoming_blocks.contains(bb));
		expected_incoming_blocks.erase(bb);

		ASSERT(!unique_values.contains(val.getImpl()));
		unique_values.insert(val.getImpl());
	}
	EXPECT_TRUE(expected_incoming_blocks.empty());
	EXPECT_EQ(unique_values.size(), 3) << "bb4 should receive 3 distinct values (not duplicates)";
}
