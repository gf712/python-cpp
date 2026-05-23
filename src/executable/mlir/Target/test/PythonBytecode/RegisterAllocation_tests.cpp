#include "Target/PythonBytecode/LinearScanRegisterAllocation.hpp"
#include "Target/PythonBytecode/RegisterAllocationLogger.hpp"

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/Dialect.hpp"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"
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
    "emitpybytecode.STORE_NAME"(%1) <{name = "input"}> : (!python.object) -> ()
    %3 = "emitpybytecode.LOAD_CONST"() <{value = 50 : ui6}> : () -> !python.object
    "emitpybytecode.STORE_NAME"(%3) <{name = "position"}> : (!python.object) -> ()
    %5 = "emitpybytecode.LOAD_NAME"() <{name = "input"}> : () -> !python.object
    %6 = "emitpybytecode.LOAD_METHOD"(%5) <{method_name = "split"}> : (!python.object) -> !python.object
    %7 = "emitpybytecode.LOAD_CONST"() <{value = "\0A"}> : () -> !python.object
    %8 = "emitpybytecode.CALL"(%6, %7) : (!python.object, !python.object) -> !python.object
    %9 = "emitpybytecode.GET_ITER"(%8) : (!python.object) -> !python.object
    cf.br ^bb1
  ^bb1:
    "emitpybytecode.FOR_ITER"(%9)[^bb2, ^bb19] : (!python.object) -> ()
  ^bb2(%10: !python.object):
    "emitpybytecode.STORE_NAME"(%10) <{name = "line"}> : (!python.object) -> ()
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
    "emitpybytecode.STORE_NAME"(%26) <{name = "move_by"}> : (!python.object) -> ()
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
    "emitpybytecode.STORE_NAME"(%35) <{name = "move_by"}> : (!python.object) -> ()
    cf.br ^bb6
  ^bb6:
    %37 = "emitpybytecode.LOAD_NAME"() <{name = "position"}> : () -> !python.object
    %38 = "emitpybytecode.LOAD_NAME"() <{name = "move_by"}> : () -> !python.object
    %39 = "emitpybytecode.INPLACE_OP"(%37, %38) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    "emitpybytecode.STORE_NAME"(%37) <{name = "position"}> : (!python.object) -> ()
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
    "emitpybytecode.STORE_NAME"(%56) <{name = "position"}> : (!python.object) -> ()
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
    "emitpybytecode.STORE_NAME"(%62) <{name = "position"}> : (!python.object) -> ()
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
	{
		return mlir::parseSourceString<mlir::ModuleOp>(FORITER_BUG_MLIR, &m_context);
	}
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
		if (block == loop_exit || loopBlocks.contains(block)) { continue; }
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
		if (loopBlocks.contains(op->getBlock())) { loopBodyLoadNames.push_back(op.getResult()); }
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

/**
 * Test that spilling works when register pressure transiently exceeds the 32-register limit.
 *
 * Pattern: one long-lived value (%x = LOAD_NAME) defined early, then 32 short-lived
 * constants (%v0..%v31), then a reduction chain of BINARY_OPs that quickly consumes
 * all the short-lived values, then a CALL using only %x.
 *
 * Without spilling: when %v31 is processed, all of %x + %v0..%v30 = 32 active, plus
 * the new %v31 makes 33 — no register available → would crash.
 *
 * With spilling: %x (furthest end) is spilled right after its definition, freeing its
 * register for the short-lived values. At peak only %v0..%v31 = 32 values are active.
 * %x is reloaded before the CALL. Each allocation pass terminates with ≤32 live values.
 */
constexpr const char *SPILL_PRESSURE_MLIR = R"(
module {
  func.func private @spill_test() -> !python.object attributes {locals = [], names = ["x"]} {
    // Long-lived value — will be spilled to make room for the 32 short-lived constants
    %x   = "emitpybytecode.LOAD_NAME"() <{name = "x"}> : () -> !python.object
    // 32 short-lived constants
    %v0  = "emitpybytecode.LOAD_CONST"() <{value = 0  : ui6}> : () -> !python.object
    %v1  = "emitpybytecode.LOAD_CONST"() <{value = 1  : ui6}> : () -> !python.object
    %v2  = "emitpybytecode.LOAD_CONST"() <{value = 2  : ui6}> : () -> !python.object
    %v3  = "emitpybytecode.LOAD_CONST"() <{value = 3  : ui6}> : () -> !python.object
    %v4  = "emitpybytecode.LOAD_CONST"() <{value = 4  : ui6}> : () -> !python.object
    %v5  = "emitpybytecode.LOAD_CONST"() <{value = 5  : ui6}> : () -> !python.object
    %v6  = "emitpybytecode.LOAD_CONST"() <{value = 6  : ui6}> : () -> !python.object
    %v7  = "emitpybytecode.LOAD_CONST"() <{value = 7  : ui6}> : () -> !python.object
    %v8  = "emitpybytecode.LOAD_CONST"() <{value = 8  : ui6}> : () -> !python.object
    %v9  = "emitpybytecode.LOAD_CONST"() <{value = 9  : ui6}> : () -> !python.object
    %v10 = "emitpybytecode.LOAD_CONST"() <{value = 10 : ui6}> : () -> !python.object
    %v11 = "emitpybytecode.LOAD_CONST"() <{value = 11 : ui6}> : () -> !python.object
    %v12 = "emitpybytecode.LOAD_CONST"() <{value = 12 : ui6}> : () -> !python.object
    %v13 = "emitpybytecode.LOAD_CONST"() <{value = 13 : ui6}> : () -> !python.object
    %v14 = "emitpybytecode.LOAD_CONST"() <{value = 14 : ui6}> : () -> !python.object
    %v15 = "emitpybytecode.LOAD_CONST"() <{value = 15 : ui6}> : () -> !python.object
    %v16 = "emitpybytecode.LOAD_CONST"() <{value = 16 : ui6}> : () -> !python.object
    %v17 = "emitpybytecode.LOAD_CONST"() <{value = 17 : ui6}> : () -> !python.object
    %v18 = "emitpybytecode.LOAD_CONST"() <{value = 18 : ui6}> : () -> !python.object
    %v19 = "emitpybytecode.LOAD_CONST"() <{value = 19 : ui6}> : () -> !python.object
    %v20 = "emitpybytecode.LOAD_CONST"() <{value = 20 : ui6}> : () -> !python.object
    %v21 = "emitpybytecode.LOAD_CONST"() <{value = 21 : ui6}> : () -> !python.object
    %v22 = "emitpybytecode.LOAD_CONST"() <{value = 22 : ui6}> : () -> !python.object
    %v23 = "emitpybytecode.LOAD_CONST"() <{value = 23 : ui6}> : () -> !python.object
    %v24 = "emitpybytecode.LOAD_CONST"() <{value = 24 : ui6}> : () -> !python.object
    %v25 = "emitpybytecode.LOAD_CONST"() <{value = 25 : ui6}> : () -> !python.object
    %v26 = "emitpybytecode.LOAD_CONST"() <{value = 26 : ui6}> : () -> !python.object
    %v27 = "emitpybytecode.LOAD_CONST"() <{value = 27 : ui6}> : () -> !python.object
    %v28 = "emitpybytecode.LOAD_CONST"() <{value = 28 : ui6}> : () -> !python.object
    %v29 = "emitpybytecode.LOAD_CONST"() <{value = 29 : ui6}> : () -> !python.object
    %v30 = "emitpybytecode.LOAD_CONST"() <{value = 30 : ui6}> : () -> !python.object
    %v31 = "emitpybytecode.LOAD_CONST"() <{value = 31 : ui6}> : () -> !python.object
    // Reduction chain: consume all %vi in a left-fold, each step frees two values and
    // produces one. By the end all short-lived values are gone.
    %r0  = "emitpybytecode.BINARY_OP"(%v0,  %v1)  <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r1  = "emitpybytecode.BINARY_OP"(%r0,  %v2)  <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r2  = "emitpybytecode.BINARY_OP"(%r1,  %v3)  <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r3  = "emitpybytecode.BINARY_OP"(%r2,  %v4)  <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r4  = "emitpybytecode.BINARY_OP"(%r3,  %v5)  <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r5  = "emitpybytecode.BINARY_OP"(%r4,  %v6)  <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r6  = "emitpybytecode.BINARY_OP"(%r5,  %v7)  <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r7  = "emitpybytecode.BINARY_OP"(%r6,  %v8)  <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r8  = "emitpybytecode.BINARY_OP"(%r7,  %v9)  <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r9  = "emitpybytecode.BINARY_OP"(%r8,  %v10) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r10 = "emitpybytecode.BINARY_OP"(%r9,  %v11) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r11 = "emitpybytecode.BINARY_OP"(%r10, %v12) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r12 = "emitpybytecode.BINARY_OP"(%r11, %v13) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r13 = "emitpybytecode.BINARY_OP"(%r12, %v14) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r14 = "emitpybytecode.BINARY_OP"(%r13, %v15) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r15 = "emitpybytecode.BINARY_OP"(%r14, %v16) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r16 = "emitpybytecode.BINARY_OP"(%r15, %v17) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r17 = "emitpybytecode.BINARY_OP"(%r16, %v18) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r18 = "emitpybytecode.BINARY_OP"(%r17, %v19) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r19 = "emitpybytecode.BINARY_OP"(%r18, %v20) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r20 = "emitpybytecode.BINARY_OP"(%r19, %v21) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r21 = "emitpybytecode.BINARY_OP"(%r20, %v22) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r22 = "emitpybytecode.BINARY_OP"(%r21, %v23) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r23 = "emitpybytecode.BINARY_OP"(%r22, %v24) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r24 = "emitpybytecode.BINARY_OP"(%r23, %v25) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r25 = "emitpybytecode.BINARY_OP"(%r24, %v26) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r26 = "emitpybytecode.BINARY_OP"(%r25, %v27) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r27 = "emitpybytecode.BINARY_OP"(%r26, %v28) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r28 = "emitpybytecode.BINARY_OP"(%r27, %v29) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r29 = "emitpybytecode.BINARY_OP"(%r28, %v30) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    %r30 = "emitpybytecode.BINARY_OP"(%r29, %v31) <{operation_type = 0 : ui8}> : (!python.object, !python.object) -> !python.object
    // Now only %x (reloaded from spill slot) and %r30 are live — well within 32 regs
    %result = "emitpybytecode.CALL"(%x, %r30) : (!python.object, !python.object) -> !python.object
    return %result : !python.object
  }
}
)";

TEST(SpillTest, SpillingHandlesTransientRegisterPressureAbove32)
{
	mlir::MLIRContext ctx;
	ctx.getOrLoadDialect<mlir::func::FuncDialect>();
	ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
	ctx.getOrLoadDialect<mlir::emitpybytecode::EmitPythonBytecodeDialect>();
	ctx.getOrLoadDialect<mlir::py::PythonDialect>();

	auto module = mlir::parseSourceString<mlir::ModuleOp>(SPILL_PRESSURE_MLIR, &ctx);
	ASSERT_TRUE(module) << "Failed to parse spill-pressure MLIR IR";

	auto funcs = module->getOps<mlir::func::FuncOp>();
	ASSERT_FALSE(funcs.empty());
	auto func = *funcs.begin();

	mlir::OpBuilder builder(module->getContext());
	LinearScanRegisterAllocation regalloc;

	// Must not crash — spilling must handle the transient pressure
	ASSERT_NO_THROW(regalloc.analyse(func, builder));

	// After the final pass every value must map to a Reg (StackLocation is only transient)
	EXPECT_FALSE(regalloc.value2mem_map.empty());
	for (const auto &[value, location] : regalloc.value2mem_map) {
		EXPECT_TRUE(std::holds_alternative<LinearScanRegisterAllocation::Reg>(location))
			<< "After the final allocation pass every value must map to a Reg, not StackLocation";
	}

	// Verify that at least one __spill_ slot was actually created
	auto locals_attr = func->getAttr("locals");
	ASSERT_TRUE(locals_attr) << "Function must have a 'locals' attribute after spilling";
	auto locals = mlir::cast<mlir::ArrayAttr>(locals_attr);
	bool found_spill_slot = false;
	for (auto attr : locals) {
		if (mlir::cast<mlir::StringAttr>(attr).getValue().starts_with("__spill_")) {
			found_spill_slot = true;
			break;
		}
	}
	EXPECT_TRUE(found_spill_slot) << "At least one __spill_N slot must have been created";
}

/**
 * Unit test for LiveInterval::alive_at() precision.
 *
 * Verifies that alive_at() correctly uses sub-interval membership rather than a
 * conservative full-span check. A value with intervals [2,5) and [8,11) must be
 * reported as dead during the gap [5,8).
 */
TEST(LiveIntervalTest, AliveAtUsesSubIntervalMembershipNotConservativeSpan)
{
	codegen::LiveIntervalAnalysis::LiveInterval interval;
	// Add two non-contiguous sub-intervals: [2,5) and [8,11)
	interval.intervals = { { 2, 5 }, { 8, 11 } };
	// Dummy value — not used by alive_at()
	interval.value = mlir::Value{};

	// Inside first sub-interval
	EXPECT_TRUE(interval.alive_at(2));
	EXPECT_TRUE(interval.alive_at(3));
	EXPECT_TRUE(interval.alive_at(4));

	// In the gap between sub-intervals — must be false (not conservative span)
	EXPECT_FALSE(interval.alive_at(5));
	EXPECT_FALSE(interval.alive_at(6));
	EXPECT_FALSE(interval.alive_at(7));

	// Inside second sub-interval
	EXPECT_TRUE(interval.alive_at(8));
	EXPECT_TRUE(interval.alive_at(9));
	EXPECT_TRUE(interval.alive_at(10));

	// Beyond the end
	EXPECT_FALSE(interval.alive_at(11));
	EXPECT_FALSE(interval.alive_at(100));
}

/**
 * Test that GET_ITER live intervals are collapsed to a single contiguous span
 * by extend_iterator_liveness().
 *
 * Parses the FOR_ITER bug IR (which has a loop), runs live interval analysis,
 * and verifies the GET_ITER value has exactly one sub-interval.
 */
TEST_F(RegisterAllocationTest, GetIterIntervalIsContiguousAfterLivenessExtension)
{
	auto module = parseForIterBugIR();
	ASSERT_TRUE(module);

	auto funcs = module->getOps<mlir::func::FuncOp>();
	ASSERT_FALSE(funcs.empty());
	auto func = *funcs.begin();

	// Run only live interval analysis (not full register allocation)
	codegen::LiveIntervalAnalysis live_analysis;
	live_analysis.analyse(func);

	// Find the GET_ITER interval
	const codegen::LiveIntervalAnalysis::LiveInterval *get_iter_interval = nullptr;
	for (const auto &interval : live_analysis.sorted_live_intervals) {
		if (!std::holds_alternative<mlir::Value>(interval.value)) { continue; }
		auto val = std::get<mlir::Value>(interval.value);
		if (val.getDefiningOp() && mlir::isa<mlir::emitpybytecode::GetIter>(val.getDefiningOp())) {
			get_iter_interval = &interval;
			break;
		}
	}

	ASSERT_NE(get_iter_interval, nullptr) << "GET_ITER interval not found";

	// After extend_iterator_liveness(), the GET_ITER must have exactly one sub-interval
	EXPECT_EQ(get_iter_interval->intervals.size(), 1u)
		<< "GET_ITER interval must be collapsed to a single contiguous span; "
		<< "found " << get_iter_interval->intervals.size() << " sub-intervals";

	// And it must be alive throughout the entire loop span
	const size_t start = get_iter_interval->start();
	const size_t end = get_iter_interval->end();
	for (size_t t = start; t < end; ++t) {
		EXPECT_TRUE(get_iter_interval->alive_at(t))
			<< "GET_ITER must be alive at every timestep in [" << start << ", " << end
			<< ") but alive_at(" << t << ") returned false";
	}
}

}// namespace
