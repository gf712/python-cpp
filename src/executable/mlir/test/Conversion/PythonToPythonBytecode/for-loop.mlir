// RUN: python-mlir-opt %s --convert-py-forloop | FileCheck %s

// Spot-check the ForLoop -> CFG + ForIter lowering. The structural
// pattern splits the parent block at the for-loop, drops in a
// GetIter for the iterable, threads ForIter ops through the body /
// step / orelse regions, and replaces py.br_yield in the loop body
// with cf.br targeting either the for-iter block (continue /
// fallthrough) or the loop's end block (break).

module {
  // CHECK-LABEL: @for_no_orelse
  func.func @for_no_orelse(%it: !python.object) {
    // CHECK: %[[ITER:.*]] = "emitpybytecode.GET_ITER"
    // The body falls through to the FOR_ITER block via a plain cf.br.
    // CHECK: cf.br ^bb{{[0-9]+}}
    // CHECK: "emitpybytecode.FOR_ITER"(%[[ITER]])
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

  // CHECK-LABEL: @for_with_break
  func.func @for_with_break(%it: !python.object) {
    // A `break` inside a for body lowers to cf.br targeting the
    // loop's end block (not the iterator-next block).
    // CHECK: "emitpybytecode.GET_ITER"
    // CHECK: "emitpybytecode.FOR_ITER"
    "python.for_loop"(%it) ({
    ^bb0:
      // kind = 1 is LoopOpKind::break_.
      "python.br_yield"() <{kind = 1 : i64}> : () -> ()
    }, {
    ^bb0(%v: !python.object):
      "python.br_yield"() : () -> ()
    }, {
    }) : (!python.object) -> ()
    return
  }
}
