include "mlir/IR/OpBase.td"

def EmitPythonBytecode_Dialect : Dialect {
  let name = "emitpybytecode";
  let cppNamespace = "::mlir::emitpybytecode";

  let summary = "Dialect to generate Python bytecode for the PythonCpp VM from MLIR.";
  let description = "TODO";
}