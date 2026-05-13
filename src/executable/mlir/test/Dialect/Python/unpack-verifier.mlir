// RUN: python-mlir-opt %s -verify-diagnostics -split-input-file

// py.unpack has variadic results, but a zero-result unpack would
// have no Python-level target receiving the values and the bytecode
// emitter would still pay for the iterator machinery. The verifier
// rejects it up front.

module {
  func.func @unpack_zero_results(%it: !python.object) {
    // expected-error @+1 {{must produce at least one unpacked value}}
    "python.unpack"(%it) : (!python.object) -> ()
    return
  }
}

// -----

module {
  func.func @unpack_two_results(%it: !python.object) -> (!python.object, !python.object) {
    %0:2 = "python.unpack"(%it) : (!python.object) -> (!python.object, !python.object)
    return %0#0, %0#1 : !python.object, !python.object
  }
}
