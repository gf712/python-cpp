// RUN: python-mlir-opt %s -verify-diagnostics -split-input-file

// py.class carries a parallel (keywords, kwargs) pair like py.call.
// Used for things like `class Foo(metaclass=Meta, total=True):` where
// each kwargs operand is named by the same-index keywords string. No
// expansion mode (no equivalent of class Foo(**kwds):), so the
// parallel rule always applies.

module {
  func.func @class_kw_mismatch(%base: !python.object, %v: !python.object) {
    // 2 names, 1 value.
    // expected-error @+1 {{has 2 keyword name(s) but 1 kwargs value(s)}}
    %0 = "python.class"(%base, %v) <{
      bases = !python.object,
      name = "Foo",
      mangled_name = "Foo",
      keywords = dense<["metaclass", "total"]> : tensor<2x!llvm.ptr>,
      captures = dense<> : tensor<0x!llvm.ptr>,
      operandSegmentSizes = array<i32: 1, 1>
    }> ({
      %r = "python.constant"() <{value}> : () -> !python.object
      "python.class_return"(%r) : (!python.object) -> ()
    }) : (!python.object, !python.object) -> !python.object
    return
  }
}

// -----

module {
  func.func @class_kw_ok(%base: !python.object, %v: !python.object) {
    %0 = "python.class"(%base, %v) <{
      name = "Foo",
      mangled_name = "Foo",
      keywords = dense<["metaclass"]> : tensor<1x!llvm.ptr>,
      captures = dense<> : tensor<0x!llvm.ptr>,
      operandSegmentSizes = array<i32: 1, 1>
    }> ({
      %r = "python.constant"() <{value}> : () -> !python.object
      "python.class_return"(%r) : (!python.object) -> ()
    }) : (!python.object, !python.object) -> !python.object
    return
  }
}
