// RUN: python-mlir-opt %s | FileCheck %s

// py.for_loop is the canonical for-loop op in the Python dialect.
// Confirms the op round-trips with its three regions (body, step,
// orelse) and a single PyObject iterable operand. The historical
// py.for_iter op was removed: the iter-then-branch lowering lives in
// the emitpybytecode dialect (emitpybytecode.FOR_ITER) and is produced
// only during the conversion pass.

module {
  // CHECK-LABEL: @for_loop_round_trip
  func.func @for_loop_round_trip(%it: !python.object) {
    // CHECK: "python.for_loop"({{.*}}) ({
    "python.for_loop"(%it) ({
    ^bb0:
      "python.br_yield"() : () -> ()
    }, {
    ^bb0(%v: !python.object):
      "python.br_yield"() : () -> ()
    }, {
    }) : (!python.object) -> ()
    return
  }
}
