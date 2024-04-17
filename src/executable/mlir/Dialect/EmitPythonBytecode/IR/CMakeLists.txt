set(LLVM_TARGET_DEFINITIONS EmitPythonBytecode.td)
mlir_tablegen(EmitPythonBytecodeDialect.h.inc -gen-dialect-decls -dialect=emitpybytecode EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/executable/mlir/Dialect)
mlir_tablegen(EmitPythonBytecodeDialect.cpp.inc -gen-dialect-defs -dialect=emitpybytecode EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/executable/mlir/Dialect)
mlir_tablegen(EmitPythonBytecodeOps.h.inc -gen-op-decls -dialect=emitpybytecode EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/executable/mlir/Dialect)
mlir_tablegen(EmitPythonBytecodeOps.cpp.inc -gen-op-defs -dialect=emitpybytecode EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/executable/mlir/Dialect)
add_public_tablegen_target(MLIREmitPythonBytecodeIncGen)

add_mlir_library(EmitPythonBytecodeDialect
                EmitPythonBytecode.cpp
                
                DEPENDS
                MLIREmitPythonBytecodeIncGen

                LINK_LIBS PUBLIC
                MLIRIR
)

target_include_directories(EmitPythonBytecodeDialect PRIVATE 
                            ${MLIR_INCLUDE_DIRS} 
                            ${CMAKE_SOURCE_DIR}/src/executable/mlir/Dialect 
                            ${CMAKE_BINARY_DIR}/src/executable/mlir/Dialect)
