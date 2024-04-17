find_package(LLVM 18.1 REQUIRED CONFIG)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR CONFIG REQUIRED
             HINTS ${LLVM_DIR}/../lib/cmake/mlir)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# enable exception handling for now..
set(LLVM_REQUIRES_EH ON)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)
include(AddLLVM)

set(PYTHON_MLIR_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/executable/mlir)
set(PYTHON_MLIR_BINARY_DIR ${PROJECT_BINARY_DIR}/src/executable/mlir)

add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)

add_library(python-mlir compile.cpp)
target_link_libraries(python-mlir PRIVATE PythonMLIRDialect TargetPythonBytecode PythonConversionPasses)