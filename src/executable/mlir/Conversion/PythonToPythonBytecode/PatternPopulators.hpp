#pragma once

#include "mlir/IR/PatternMatch.h"

namespace mlir::py {

// Per-op-family pattern populators. PythonToPythonBytecodePass calls
// each populator to assemble its full RewritePatternSet. Splitting
// these out keeps the monolithic conversion file from ballooning past
// the point where it can be reviewed in one sitting.

void populateArithPatterns(mlir::RewritePatternSet &patterns);
void populateAttributeSubscriptPatterns(mlir::RewritePatternSet &patterns);
void populateCollectionPatterns(mlir::RewritePatternSet &patterns);
void populateControlFlowPatterns(mlir::RewritePatternSet &patterns);
void populateFunctionPatterns(mlir::RewritePatternSet &patterns);
void populateImportPatterns(mlir::RewritePatternSet &patterns);
void populateLoadStorePatterns(mlir::RewritePatternSet &patterns);

}// namespace mlir::py
