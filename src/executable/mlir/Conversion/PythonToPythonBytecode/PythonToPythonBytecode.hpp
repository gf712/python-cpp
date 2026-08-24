#pragma once

#include <memory>

namespace mlir {

class Pass;

namespace py {
	std::unique_ptr<Pass> createPythonToPythonBytecodePass();

	// Dedicated passes for the region-bearing control-flow ops. Each runs only its
	// own lowering pattern(s) and is meant to slot into the pipeline ahead of the
	// monolithic conversion pass, so that canonicalize / CSE can be interleaved
	// between them. Plan step 18.
	//
	// Both loops share one pass: their patterns defer to each other so that a
	// nested loop is flattened before the enclosing one retargets the break /
	// continue in its orelse, and that handshake needs both patterns in the same
	// greedy run. createConvertForLoopPass/createConvertWhileLoopPass remain for
	// python-mlir-opt, which drives a single lowering at a time.
	std::unique_ptr<Pass> createConvertLoopsPass();
	std::unique_ptr<Pass> createConvertForLoopPass();
	std::unique_ptr<Pass> createConvertWhileLoopPass();
	std::unique_ptr<Pass> createConvertTryPass();
	std::unique_ptr<Pass> createConvertWithPass();

	// Materialise an emitpybytecode constant None as the operand of any
	// zero-operand func.return inside a func.func whose result type is
	// non-empty. Used as a follow-up to mlir::createRemoveDeadValuesPass,
	// which can strip the return's operand when its producer becomes
	// dead, leaving zero-operand returns that violate the bytecode
	// emitter's exactly-one-operand invariant.
	std::unique_ptr<Pass> createMaterialiseReturnNonePass();
}// namespace py

}// namespace mlir
