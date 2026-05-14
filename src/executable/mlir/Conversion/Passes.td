include "mlir/Pass/PassBase.td"


def ConvertPythonToPythonBytecode : Pass<"convert-python-to-pythonbytecode"> {
  let summary = "Convert recognized Python ops to PythonCpp bytecode";
  let constructor = "mlir::py::createPythonToPythonBytecodePass()";
}

def ConvertPyForLoop : Pass<"convert-py-forloop"> {
  let summary = "Lower py.for_loop to emitpybytecode control flow";
  let constructor = "mlir::py::createConvertForLoopPass()";
}

def ConvertPyWhile : Pass<"convert-py-while"> {
  let summary = "Lower py.while to emitpybytecode control flow";
  let constructor = "mlir::py::createConvertWhileLoopPass()";
}

def ConvertPyTry : Pass<"convert-py-try"> {
  let summary = "Lower py.try (and its handler scopes) to emitpybytecode control flow";
  let constructor = "mlir::py::createConvertTryPass()";
}

def ConvertPyWith : Pass<"convert-py-with"> {
  let summary = "Lower py.with to emitpybytecode control flow";
  let constructor = "mlir::py::createConvertWithPass()";
}

def MaterialiseReturnNone : Pass<"materialise-return-none", "::mlir::func::FuncOp"> {
  let summary = "Insert emitpybytecode.LOAD_CONST(None) before zero-operand func.return ops";
  let constructor = "mlir::py::createMaterialiseReturnNonePass()";
  let dependentDialects = ["::mlir::emitpybytecode::EmitPythonBytecodeDialect"];
}