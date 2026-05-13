// RUN: python-mlir-opt %s --python-to-pythonbytecode | FileCheck %s

// py.binary lowers to emitpybytecode.BINARY_OP with the kind enum
// translated to the bytecode-level Operation enum (PLUS=0, MINUS=1,
// ..., MATMUL=12). Spot-check the mapping for a few representative
// kinds.

module {
  // CHECK-LABEL: @lower_add
  func.func @lower_add(%a: !python.object {llvm.name = "a"}, %b: !python.object {llvm.name = "b"}) -> !python.object {
    // CHECK: "emitpybytecode.BINARY_OP"({{.*}}) <{operation_type = 0 : ui8}>
    %0 = "python.binary"(%a, %b) {kind = 0 : i64} : (!python.object, !python.object) -> !python.object
    return %0 : !python.object
  }

  // CHECK-LABEL: @lower_matmul
  func.func @lower_matmul(%a: !python.object {llvm.name = "a"}, %b: !python.object {llvm.name = "b"}) -> !python.object {
    // BinaryOpKind::mmul=7 maps to BinaryOperation::Operation::MATMUL=12.
    // CHECK: "emitpybytecode.BINARY_OP"({{.*}}) <{operation_type = 12 : ui8}>
    %0 = "python.binary"(%a, %b) {kind = 7 : i64} : (!python.object, !python.object) -> !python.object
    return %0 : !python.object
  }

  // CHECK-LABEL: @lower_xor
  func.func @lower_xor(%a: !python.object {llvm.name = "a"}, %b: !python.object {llvm.name = "b"}) -> !python.object {
    // BinaryOpKind::xor_=12 maps to BinaryOperation::Operation::XOR=11.
    // CHECK: "emitpybytecode.BINARY_OP"({{.*}}) <{operation_type = 11 : ui8}>
    %0 = "python.binary"(%a, %b) {kind = 12 : i64} : (!python.object, !python.object) -> !python.object
    return %0 : !python.object
  }
}
