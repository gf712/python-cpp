add_mlir_translation_library(TargetPythonBytecode
  TranslateToPythonBytecode.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/src/executable/mlir/

  LINK_LIBS
    PUBLIC
      EmitPythonBytecodeDialect
      MLIRIR
      spdlog
    PRIVATE
      python-cpp
      MLIRTargetLLVMIRExport
)

target_include_directories(TargetPythonBytecode
                           SYSTEM
                           PUBLIC
                           ${MLIR_INCLUDE_DIRS}
)


target_include_directories(TargetPythonBytecode
                           PUBLIC
                           ${CMAKE_SOURCE_DIR}/src/executable/mlir/
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/src/executable/mlir/Dialect
                           ${CMAKE_BINARY_DIR}/src/executable/mlir/Dialect
)

#                             ${CMAKE_SOURCE_DIR}/src
#                             ${CMAKE_SOURCE_DIR}/src/executable/mlir/include/mlir/Dialect
#                             ${CMAKE_BINARY_DIR}/src/executable/mlir/include/mlir/Dialect
#                             ${MLIR_INCLUDE_DIRS}
#                             )

