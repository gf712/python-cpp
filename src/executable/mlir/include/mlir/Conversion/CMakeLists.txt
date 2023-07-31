set(LLVM_TARGET_DEFINITIONS Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS})
add_public_tablegen_target(PythonConversionPassIncGen)