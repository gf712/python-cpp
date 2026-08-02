# LLVMConfigVersion.cmake only accepts an exact major.minor match, so asking for
# "23" means "23.0" and rejects the 23.1.x releases. Search without a version,
# newest install first, and enforce the major version ourselves.
set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)
find_package(LLVM REQUIRED CONFIG)
if(LLVM_VERSION_MAJOR LESS 23)
  message(FATAL_ERROR
          "LLVM 23 or newer is required, found ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}. "
          "Point CMake at a newer install with -DLLVM_DIR=<prefix>/lib/cmake/llvm")
endif()
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR} (${LLVM_PACKAGE_VERSION})")

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
add_subdirectory(tools/python-mlir-opt)
add_subdirectory(test)

add_library(python-mlir compile.cpp)
target_link_libraries(python-mlir PRIVATE PythonMLIRDialect TargetPythonBytecode PythonConversionPasses)
