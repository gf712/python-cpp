// RUN: python-mlir-opt %s | FileCheck %s

// py.build_slice.step is Optional<>: a Python slice like a[1:5]
// (no step) lowers without materializing a None constant for step,
// saving a register relative to the historical 3-operand-always
// representation. lower and upper are still required because the
// runtime BuildSlice::execute dereferences both unconditionally.

module {
  // CHECK-LABEL: @slice_with_step
  func.func @slice_with_step(%lo: !python.object, %hi: !python.object, %st: !python.object) -> !python.object {
    // CHECK: "python.build_slice"({{%[a-z0-9]+, %[a-z0-9]+, %[a-z0-9]+}})
    %0 = "python.build_slice"(%lo, %hi, %st) : (!python.object, !python.object, !python.object) -> !python.object
    return %0 : !python.object
  }

  // CHECK-LABEL: @slice_no_step
  func.func @slice_no_step(%lo: !python.object, %hi: !python.object) -> !python.object {
    // Without the step operand the op survives parse / print.
    // CHECK: "python.build_slice"({{%[a-z0-9]+, %[a-z0-9]+}})
    // CHECK-NOT: %[a-z0-9]+, %[a-z0-9]+, %[a-z0-9]+
    %0 = "python.build_slice"(%lo, %hi) : (!python.object, !python.object) -> !python.object
    return %0 : !python.object
  }
}
