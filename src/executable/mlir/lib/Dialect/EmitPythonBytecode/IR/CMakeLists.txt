add_mlir_library(EmitPythonBytecodeDialect
                EmitPythonBytecode.cpp
                
                DEPENDS
                MLIREmitPythonBytecodeIncGen

                LINK_LIBS PUBLIC
                MLIRIR
)

target_include_directories(EmitPythonBytecodeDialect PRIVATE 
                            ${MLIR_INCLUDE_DIRS} 
                            ${CMAKE_SOURCE_DIR}/src/executable/mlir/include/mlir/Dialect 
                            ${CMAKE_BINARY_DIR}/src/executable/mlir/include/mlir/Dialect)
