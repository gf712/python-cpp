add_mlir_translation_library(TargetPythonBytecode
  TranslateToPythonBytecode.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/src/executable/mlir/include/mlir/

  LINK_LIBS 
    PUBLIC
      EmitPythonBytecodeDialect
      MLIRIR
    PRIVATE
      spdlog
      python-cpp
      MLIRTargetLLVMIRExport
)

target_include_directories(TargetPythonBytecode 
                           PUBLIC 
                           ${CMAKE_SOURCE_DIR}/src/executable/mlir/include/mlir/
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/src/executable/mlir/include/mlir/Dialect
                           ${CMAKE_BINARY_DIR}/src/executable/mlir/include/mlir/Dialect
                           ${MLIR_INCLUDE_DIRS}
)

#                             ${CMAKE_SOURCE_DIR}/src
#                             ${CMAKE_SOURCE_DIR}/src/executable/mlir/include/mlir/Dialect
#                             ${CMAKE_BINARY_DIR}/src/executable/mlir/include/mlir/Dialect
#                             ${MLIR_INCLUDE_DIRS}
#                             )

