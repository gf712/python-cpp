// RUN: python-mlir-opt %s -verify-diagnostics -split-input-file

// py.call carries a parallel (keywords, kwargs) pair: a dense string
// array of keyword names alongside a variadic operand list of values.
// The two must have the same length so each value lines up with one
// name. Previously asserted at lowering time; the verifier catches
// the mismatch up front.

module {
  func.func @kw_count_mismatch(%c: !python.object, %v: !python.object) -> !python.object {
    // 2 names, 1 value.
    // expected-error @+1 {{has 2 keyword name(s) but 1 kwargs value(s)}}
    %0 = "python.call"(%c, %v) <{
      keywords = dense<["a", "b"]> : tensor<2x!llvm.ptr>,
      operandSegmentSizes = array<i32: 1, 0, 1>,
      requires_args_expansion = false,
      requires_kwargs_expansion = false
    }> : (!python.object, !python.object) -> !python.object
    return %0 : !python.object
  }
}

// -----

module {
  func.func @kw_count_ok(%c: !python.object, %v: !python.object) -> !python.object {
    // Matched pair: 1 name, 1 value. Should verify cleanly.
    %0 = "python.call"(%c, %v) <{
      keywords = dense<["a"]> : tensor<1x!llvm.ptr>,
      operandSegmentSizes = array<i32: 1, 0, 1>,
      requires_args_expansion = false,
      requires_kwargs_expansion = false
    }> : (!python.object, !python.object) -> !python.object
    return %0 : !python.object
  }
}

// -----

module {
  func.func @kw_expansion_ok(%c: !python.object, %d: !python.object) -> !python.object {
    // **kwds-style call: requires_kwargs_expansion=true so the single
    // kwargs operand is a dict to expand, and keywords is empty by
    // construction. The verifier short-circuits the keywords/kwargs
    // size check in this case.
    %0 = "python.call"(%c, %d) <{
      keywords = dense<> : tensor<0x!llvm.ptr>,
      operandSegmentSizes = array<i32: 1, 0, 1>,
      requires_args_expansion = false,
      requires_kwargs_expansion = true
    }> : (!python.object, !python.object) -> !python.object
    return %0 : !python.object
  }
}

// -----

module {
  func.func @kw_expansion_too_many(%c: !python.object, %d1: !python.object, %d2: !python.object) -> !python.object {
    // requires_kwargs_expansion accepts at most one kwargs operand
    // (the dict to expand). Two operands is a malformed call.
    // expected-error @+1 {{requires_kwargs_expansion expects at most one kwargs operand, got 2}}
    %0 = "python.call"(%c, %d1, %d2) <{
      keywords = dense<> : tensor<0x!llvm.ptr>,
      operandSegmentSizes = array<i32: 1, 0, 2>,
      requires_args_expansion = false,
      requires_kwargs_expansion = true
    }> : (!python.object, !python.object, !python.object) -> !python.object
    return %0 : !python.object
  }
}
