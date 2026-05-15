// RUN: python-mlir-opt %s | FileCheck %s

// Smoke test confirming the lit harness can load the Python dialect
// and a single op survives parse / print / re-parse. Uses generic
// op syntax because most Python_Op records don't define a custom
// assemblyFormat.

module {
  // CHECK-LABEL: @smoke_constant
  func.func @smoke_constant() -> !python.object {
    // CHECK: "python.constant"() <{value = 42 : i64}> : () -> !python.object
    %0 = "python.constant"() {value = 42 : i64} : () -> !python.object
    return %0 : !python.object
  }
}
