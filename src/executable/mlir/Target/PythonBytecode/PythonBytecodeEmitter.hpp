#pragma once

// Names ::Program, which py.runtime owns, so this header must be included
// after `import py.runtime;` rather than forward-declaring it here: a
// global-module `class Program;` would be a different entity from the
// module-attached one.

// mlir::Operation likewise comes from the includer's own MLIR headers rather
// than a forward declaration here.

namespace codegen {
std::shared_ptr<Program> translateToPythonBytecode(mlir::Operation *op);
}
