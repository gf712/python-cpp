include "mlir/IR/OpBase.td"

def EmitPythonBytecode_Dialect : Dialect {
  let name = "emitpybytecode";
  let cppNamespace = "::mlir::emitpybytecode";

  let summary = "Dialect to generate Python bytecode for the PythonCpp VM from MLIR.";
  let description = [{
The EmitPythonBytecode dialect is the lower-level half of the
compilation pipeline. It models the wire format consumed by the
PythonCpp register-based VM as MLIR ops: each op corresponds 1:1
(or close to it) to a bytecode instruction emitted by
`codegen::translateToPythonBytecode`. Ops mirror CPython opcode
names (LOAD_FAST, BINARY_OP, CALL, …) in uppercase to make the
correspondence obvious when reading IR dumps.

Compared to the upstream `python` dialect, ops here are
deliberately denormalised: enum kinds are encoded as raw
`ui8`/`ui16` attributes matching the VM wire format, control flow
is expressed as CFG branches (no region-bearing structured ops),
and most "may run user code" semantic refinement has already
happened. As a result, this dialect is not a useful target for
high-level optimisation — its job is to provide a stable IR
checkpoint right before bytecode emission so that
canonicalize / CSE can dedupe LOAD_CONST and similar pure ops
introduced during conversion, and so that the bytecode emitter
itself is a structurally simple walker.

New ops belong here when they directly model a VM instruction
that doesn't already exist; semantic refinement (operator
dispatch, scope-aware name resolution, etc.) belongs in the
`python` dialect instead.
  }];
}