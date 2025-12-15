#include "Target/PythonBytecode/LinearScanRegisterAllocation.hpp"
#include "Target/PythonBytecode/RegisterAllocationLogger.hpp"

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/Dialect.hpp"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"
#include <llvm-20/llvm/Support/raw_ostream.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <spdlog/spdlog.h>

using namespace codegen;

namespace {

// MLIR IR from integration/minimal_foriter_bug.py
// This is the actual IR that triggers the FOR_ITER iterator clobbering bug
constexpr const char *FORITER_BUG_MLIR = R"(
module attributes {llvm.argv = ["integration/minimal_foriter_bug.py"]} {
  func.func private @__hidden_init__() -> !python.object attributes {names = ["split"]} {
    %0 = "emitpybytecode.LOAD_CONST"() <{value = "\0AMinimal reproducer for the FOR_ITER iterator clobbering bug.\0A\0AThis is the minimal case that triggers the bug:\0A- FOR loop over a method call result (.split())\0A- Complex loop body with nested while loop\0A- Multiple LOAD_NAME operations for global variables\0A- Enough register pressure that the iterator register gets reused\0A\0ABug manifests as: TypeError: 'int' object is not an iterator\0A"}> : () -> !python.object
    %1 = "emitpybytecode.LOAD_CONST"() <{value = "L68\0AL30\0AR48"}> : () -> !python.object
    %2 = "emitpybytecode.STORE_NAME"(%1) <{name = "input"}> : (!python.object) -> !python.object
    %3 = "emitpybytecode.LOAD_CONST"() <{value = 50 : ui6}> : () -> !python.object
    %4 = "emitpybytecode.STORE_NAME"(%3) <{name = "position"}> : (!python.object) -> !python.object
    %5 = "emitpybytecode.LOAD_NAME"() <{name = "input"}> : () -> !python.object
    %6 = "emitpybytecode.LOAD_METHOD"(%5) <{method_name = "split"}> : (!python.object) -> !python.object
    %7 = "emitpybytecode.LOAD_CONST"() <{value = "\0A"}> : () -> !python.object
    %8 = "emitpybytecode.CALL"(%6, %7) : (!python.object, !python.object) -> !python.object
    %9 = "emitpybytecode.GET_ITER"(%8) : (!python.object) -> !python.object
    cf.br ^bb1
  ^bb1:
    "emitpybytecode.FOR_ITER"(%9)[^bb2, ^bb19] : (!python.object) -> ()
  ^bb2(%10: !python.object):
    %11 = "emitpybytecode.STORE_NAME"(%10) <{name = "line"}> : (!python.object) -> !python.object
    cf.br ^bb3
  ^bb3:
    %12 = "emitpybytecode.LOAD_NAME"() <{name = "line"}> : () -> !python.object
    %13 = "emitpybytecode.LOAD_CONST"() <{value = 0 : ui1}> : () -> !python.object
    %14 = "emitpybytecode.BINARY_SUBSCRIPT"(%12, %13) : (!python.object, !python.object) -> !python.object
    %15 = "emitpybytecode.LOAD_CONST"() <{value = "L"}> : () -> !python.object
    %16 = "emitpybytecode.COMPARE"(%14, %15) <{predicate = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %17 = "emitpybytecode.TO_BOOL"(%16) : (!python.object) -> !python.object
    emitpybytecode.JUMP_IF_FALSE %16, ^bb4, ^bb5
  ^bb4:
    %18 = "emitpybytecode.LOAD_NAME"() <{name = "int"}> : () -> !python.object
    %19 = "emitpybytecode.LOAD_NAME"() <{name = "line"}> : () -> !python.object
    %20 = "emitpybytecode.LOAD_CONST"() <{value = 1 : ui1}> : () -> !python.object
    %21 = "emitpybytecode.LOAD_CONST"() <{value}> : () -> !python.object
    %22 = "emitpybytecode.LOAD_CONST"() <{value}> : () -> !python.object
    %23 = "emitpybytecode.BUILD_SLICE"(%20, %21, %22) : (!python.object, !python.object, !python.object) -> !python.object
    %24 = "emitpybytecode.BINARY_SUBSCRIPT"(%19, %23) : (!python.object, !python.object) -> !python.object
    %25 = "emitpybytecode.CALL"(%18, %24) : (!python.object, !python.object) -> !python.object
    %26 = "emitpybytecode.UNARY"(%25) <{operation_type = 1 : ui8}> : (!python.object) -> !python.object
    %27 = "emitpybytecode.STORE_NAME"(%26) <{name = "move_by"}> : (!python.object) -> !python.object
    cf.br ^bb6
  ^bb5:
    %28 = "emitpybytecode.LOAD_NAME"() <{name = "int"}> : () -> !python.object
    %29 = "emitpybytecode.LOAD_NAME"() <{name = "line"}> : () -> !python.object
    %30 = "emitpybytecode.LOAD_CONST"() <{value = 1 : ui1}> : () -> !python.object
    %31 = "emitpybytecode.LOAD_CONST"() <{value}> : () -> !python.object
    %32 = "emitpybytecode.LOAD_CONST"() <{value}> : () -> !python.object
    %33 = "emitpybytecode.BUILD_SLICE"(%30, %31, %32) : (!python.object, !python.object, !python.object) -> !python.object
    %34 = "emitpybytecode.BINARY_SUBSCRIPT"(%29, %33) : (!python.object, !python.object) -> !python.object
    %35 = "emitpybytecode.CALL"(%28, %34) : (!python.object, !python.object) -> !python.object
    %36 = "emitpybytecode.STORE_NAME"(%35) <{name = "move_by"}> : (!python.object) -> !python.object
    cf.br ^bb6
  ^bb6:
    %37 = "emitpybytecode.LOAD_NAME"() <{name = "position"}> : () -> !python.object
    %38 = "emitpybytecode.LOAD_NAME"() <{name = "move_by"}> : () -> !python.object
    %39 = "emitpybytecode.INPLACE_OP"(%37, %38) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %40 = "emitpybytecode.STORE_NAME"(%37) <{name = "position"}> : (!python.object) -> !python.object
    cf.br ^bb7
  ^bb7:
    cf.br ^bb8
  ^bb8:
    %41 = "emitpybytecode.LOAD_NAME"() <{name = "position"}> : () -> !python.object
    %42 = "emitpybytecode.LOAD_CONST"() <{value = 0 : ui1}> : () -> !python.object
    %43 = "emitpybytecode.COMPARE"(%41, %42) <{predicate = 2 : ui8}> : (!python.object, !python.object) -> !python.object
    %44 = "emitpybytecode.TO_BOOL"(%43) : (!python.object) -> !python.object
    emitpybytecode.JUMP_IF_FALSE %43, ^bb10(%43 : !python.object), ^bb9
  ^bb9:
    %45 = "emitpybytecode.LOAD_NAME"() <{name = "position"}> : () -> !python.object
    %46 = "emitpybytecode.LOAD_CONST"() <{value = 99 : ui7}> : () -> !python.object
    %47 = "emitpybytecode.COMPARE"(%45, %46) <{predicate = 4 : ui8}> : (!python.object, !python.object) -> !python.object
    cf.br ^bb10(%47 : !python.object)
  ^bb10(%48: !python.object):
    %49 = "emitpybytecode.TO_BOOL"(%48) : (!python.object) -> !python.object
    emitpybytecode.JUMP_IF_FALSE %48, ^bb11, ^bb17
  ^bb11:
    %50 = "emitpybytecode.LOAD_NAME"() <{name = "position"}> : () -> !python.object
    %51 = "emitpybytecode.LOAD_CONST"() <{value = 0 : ui1}> : () -> !python.object
    %52 = "emitpybytecode.COMPARE"(%50, %51) <{predicate = 2 : ui8}> : (!python.object, !python.object) -> !python.object
    %53 = "emitpybytecode.TO_BOOL"(%52) : (!python.object) -> !python.object
    emitpybytecode.JUMP_IF_FALSE %52, ^bb12, ^bb13
  ^bb12:
    %54 = "emitpybytecode.LOAD_CONST"() <{value = 100 : ui7}> : () -> !python.object
    %55 = "emitpybytecode.LOAD_NAME"() <{name = "position"}> : () -> !python.object
    %56 = "emitpybytecode.BINARY_OP"(%54, %55) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %57 = "emitpybytecode.STORE_NAME"(%56) <{name = "position"}> : (!python.object) -> !python.object
    cf.br ^bb14
  ^bb13:
    %58 = "emitpybytecode.LOAD_NAME"() <{name = "position"}> : () -> !python.object
    %59 = "emitpybytecode.LOAD_CONST"() <{value = 99 : ui7}> : () -> !python.object
    %60 = "emitpybytecode.COMPARE"(%58, %59) <{predicate = 4 : ui8}> : (!python.object, !python.object) -> !python.object
    %61 = "emitpybytecode.TO_BOOL"(%60) : (!python.object) -> !python.object
    emitpybytecode.JUMP_IF_FALSE %60, ^bb15, ^bb16
  ^bb14:
    cf.br ^bb7
  ^bb15:
    %62 = "emitpybytecode.LOAD_NAME"() <{name = "position"}> : () -> !python.object
    %63 = "emitpybytecode.LOAD_CONST"() <{value = 100 : ui7}> : () -> !python.object
    %64 = "emitpybytecode.INPLACE_OP"(%62, %63) <{operation_type = 1 : ui8}> : (!python.object, !python.object) -> !python.object
    %65 = "emitpybytecode.STORE_NAME"(%62) <{name = "position"}> : (!python.object) -> !python.object
    cf.br ^bb16
  ^bb16:
    cf.br ^bb14
  ^bb17:
    cf.br ^bb18
  ^bb18:
    "emitpybytecode.FOR_ITER"(%9)[^bb2, ^bb19] : (!python.object) -> ()
  ^bb19:
    %66 = "emitpybytecode.LOAD_NAME"() <{name = "print"}> : () -> !python.object
    %67 = "emitpybytecode.LOAD_NAME"() <{name = "position"}> : () -> !python.object
    %68 = "emitpybytecode.CALL"(%66, %67) : (!python.object, !python.object) -> !python.object
    cf.br ^bb20
  ^bb20:
    %69 = "emitpybytecode.LOAD_CONST"() <{value}> : () -> !python.object
    return %69 : !python.object
  }
}
)";

class RegisterAllocationTest : public ::testing::Test
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
		logger->set_level(spdlog::level::warn);
	}

	// Parse the MLIR IR that reproduces the FOR_ITER bug
	mlir::OwningOpRef<mlir::ModuleOp> parseForIterBugIR()
	{ return mlir::parseSourceString<mlir::ModuleOp>(FORITER_BUG_MLIR, &m_context); }
};

/**
 * This test demonstrates the FOR_ITER iterator clobbering bug.
 *
 * The bug: When register allocation processes a FOR loop, it allocates the iterator
 * to some register (e.g., r2). Inside the loop body, when loading global variables
 * (LOAD_NAME operations), the register allocator may reuse the iterator's register
 * because it doesn't properly track that the iterator must stay alive for the entire
 * loop duration.
 *
 * This test EXPECTS TO FAIL until the bug is fixed. When the bug is present, at least
 * one LOAD_NAME operation inside the loop will be assigned the same register as the
 * iterator, causing the iterator to be clobbered.
 */
TEST_F(RegisterAllocationTest, ForIterIteratorRegisterNotReusedInLoopBody)
{
	// Parse the real MLIR IR from minimal_foriter_bug.py
	auto module = parseForIterBugIR();
	ASSERT_TRUE(module) << "Failed to parse MLIR IR";

	// Get the function
	auto funcs = module->getOps<mlir::func::FuncOp>();
	ASSERT_FALSE(funcs.empty()) << "No functions found in module";
	auto func = *funcs.begin();

	// Run register allocation
	mlir::OpBuilder builder(func->getContext());
	LinearScanRegisterAllocation regalloc;
	regalloc.analyse(func, builder);

	// Find the iterator (GET_ITER result - this is %9 in the IR)
	mlir::Value iterator;
	func.walk([&](mlir::emitpybytecode::GetIter op) {
		iterator = op.getResult();
		return mlir::WalkResult::interrupt();
	});
	ASSERT_TRUE(iterator) << "Failed to find GET_ITER operation";

	// Get the register assigned to the iterator
	auto iteratorLoc = regalloc.value2mem_map.find(iterator);
	ASSERT_NE(iteratorLoc, regalloc.value2mem_map.end()) << "Iterator has no register allocation";
	ASSERT_TRUE(std::holds_alternative<LinearScanRegisterAllocation::Reg>(iteratorLoc->second))
		<< "Iterator is not allocated to a register";
	auto iteratorReg = std::get<LinearScanRegisterAllocation::Reg>(iteratorLoc->second).idx;

	auto for_iter = mlir::dyn_cast<mlir::emitpybytecode::ForIter>(*iterator.getUsers().begin());
	ASSERT_TRUE(for_iter);

	auto loop_body = for_iter.getBody();
	auto loop_exit = for_iter.getContinuation();

	// Collect all blocks that are part of the loop (reachable from loop body but not the exit)
	llvm::SmallPtrSet<mlir::Block *, 16> loopBlocks;
	llvm::SmallVector<mlir::Block *, 8> worklist;
	worklist.push_back(loop_body);

	while (!worklist.empty()) {
		auto *block = worklist.pop_back_val();
		if (block == loop_exit || loopBlocks.contains(block)) {
			continue;
		}
		loopBlocks.insert(block);

		// Add successors to worklist
		for (auto *successor : block->getSuccessors()) {
			if (successor != loop_exit && !loopBlocks.contains(successor)) {
				worklist.push_back(successor);
			}
		}
	}

	// Find all LOAD_NAME operations inside the FOR loop body only
	// The loop body includes all blocks between the entry and exit
	// These should NOT reuse the iterator register since the iterator is still alive
	llvm::SmallVector<mlir::Value, 8> loopBodyLoadNames;

	func.walk([&](mlir::emitpybytecode::LoadNameOp op) {
		// Only collect LOAD_NAME operations that are in loop body blocks
		if (loopBlocks.contains(op->getBlock())) {
			loopBodyLoadNames.push_back(op.getResult());
		}
	});

	ASSERT_FALSE(loopBodyLoadNames.empty()) << "No LOAD_NAME operations found in loop body";

	// Check that NONE of the loop body LOAD_NAME operations reuse the iterator register
	// THIS IS THE BUG TEST: Currently this WILL fail because the register allocator
	// reuses the iterator register for LOAD_NAME operations
	int clobberedCount = 0;
	for (auto val : loopBodyLoadNames) {
		auto valLoc = regalloc.value2mem_map.find(val);
		if (valLoc != regalloc.value2mem_map.end()
			&& std::holds_alternative<LinearScanRegisterAllocation::Reg>(valLoc->second)) {
			auto valReg = std::get<LinearScanRegisterAllocation::Reg>(valLoc->second).idx;

			if (valReg == iteratorReg) {
				clobberedCount++;
				llvm::outs() << val << " clobbers iterator\n";
			}
		}
	}

	// THIS EXPECTATION WILL FAIL when the bug is present
	// When fixed, clobberedCount should be 0
	EXPECT_EQ(clobberedCount, 0) << "BUG DETECTED: " << clobberedCount
								 << " LOAD_NAME operation(s) reuse the iterator register r"
								 << iteratorReg << " - the iterator will be clobbered!";
}

/**
 * Smoke test to verify register allocation runs without crashing
 */
TEST_F(RegisterAllocationTest, RegisterAllocationRunsWithoutCrashing)
{
	auto module = parseForIterBugIR();
	ASSERT_TRUE(module);

	auto funcs = module->getOps<mlir::func::FuncOp>();
	ASSERT_FALSE(funcs.empty());
	auto func = *funcs.begin();

	mlir::OpBuilder builder(module->getContext());
	LinearScanRegisterAllocation regalloc;

	// Should not crash
	EXPECT_NO_THROW(regalloc.analyse(func, builder));

	// Should have allocated registers for some values
	EXPECT_FALSE(regalloc.value2mem_map.empty());
}

}// namespace
