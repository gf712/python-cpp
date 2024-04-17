set(LLVM_TARGET_DEFINITIONS PythonBase.td)
mlir_tablegen(PythonOpsEnums.h.inc -gen-enum-decls -dialect=python EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/executable/mlir/Dialect)
mlir_tablegen(PythonOpsEnums.cpp.inc -gen-enum-defs -dialect=python EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/executable/mlir/Dialect)

set(LLVM_TARGET_DEFINITIONS PythonOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=python EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/executable/mlir/Dialect)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=python EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/executable/mlir/Dialect)
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=python EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/executable/mlir/Dialect)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=python EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/executable/mlir/Dialect)

set(LLVM_TARGET_DEFINITIONS PythonTypes.td)
mlir_tablegen(PythonTypes.h.inc -gen-typedef-decls -dialect=python EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/executable/mlir/Dialect)
mlir_tablegen(PythonTypes.cpp.inc -gen-typedef-defs -dialect=python EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/executable/mlir/Dialect)
add_public_tablegen_target(MLIRPythonOpsIncGen)