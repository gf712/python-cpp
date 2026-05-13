// RUN: python-mlir-opt %s --python-to-pythonbytecode | FileCheck %s

// Spot-check the per-namespace template lowerings for the
// Load/Store/Delete{Name,Fast,Global,Deref} family. Each maps 1:1 to
// the corresponding emitpybytecode op while also registering the
// name on the FuncOp's "names" attribute so the bytecode emitter can
// look it up later.

module {
  // CHECK-LABEL: @lower_load_fast
  // Local x is registered on the func's "locals" alongside the arg name.
  // CHECK-SAME: attributes {locals = ["x"]}
  func.func @lower_load_fast(%a: !python.object {llvm.name = "x"}) -> !python.object {
    // CHECK: emitpybytecode.LOAD_FAST
    %0 = "python.load_fast"() {name = "x"} : () -> !python.object
    return %0 : !python.object
  }

  // CHECK-LABEL: @lower_store_global
  // CHECK-SAME: attributes {locals = ["arg"], names = ["g"]}
  func.func @lower_store_global(%v: !python.object {llvm.name = "arg"}) -> !python.object {
    // py.store_* / emitpybytecode.STORE_* are side-effecting writes that
    // do NOT produce an SSA result. The lowered op uses `-> ()`, which
    // matters for register allocation: a phantom unused result used to
    // burn a register per store.
    // CHECK: emitpybytecode.STORE_GLOBAL
    // CHECK-NOT: = "emitpybytecode.STORE_GLOBAL"
    "python.store_global"(%v) {name = "g"} : (!python.object) -> ()
    return %v : !python.object
  }
}
