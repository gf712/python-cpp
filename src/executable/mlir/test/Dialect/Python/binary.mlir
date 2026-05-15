// RUN: python-mlir-opt %s | FileCheck %s

// Round-trip the unified py.binary op across all 13 kinds. Confirms
// the Python_ArithOpKindAttr enum prints/parses and the op accepts
// every kind value (arithmetic + bitwise/logical). The same attr is
// reused by py.inplace_op, so adding a new kind here lights up both
// dialect ops.

module {
  // CHECK-LABEL: @each_kind
  func.func @each_kind(%a: !python.object, %b: !python.object) -> !python.object {
    // CHECK: "python.binary"({{.*}}) <{kind = 0 : i64}>
    %add = "python.binary"(%a, %b) {kind = 0 : i64} : (!python.object, !python.object) -> !python.object
    // CHECK: "python.binary"({{.*}}) <{kind = 1 : i64}>
    %sub = "python.binary"(%add, %b) {kind = 1 : i64} : (!python.object, !python.object) -> !python.object
    // CHECK: "python.binary"({{.*}}) <{kind = 7 : i64}>
    %matmul = "python.binary"(%sub, %b) {kind = 7 : i64} : (!python.object, !python.object) -> !python.object
    // CHECK: "python.binary"({{.*}}) <{kind = 10 : i64}>
    %and = "python.binary"(%matmul, %b) {kind = 10 : i64} : (!python.object, !python.object) -> !python.object
    // CHECK: "python.binary"({{.*}}) <{kind = 12 : i64}>
    %xor = "python.binary"(%and, %b) {kind = 12 : i64} : (!python.object, !python.object) -> !python.object
    return %xor : !python.object
  }
}
