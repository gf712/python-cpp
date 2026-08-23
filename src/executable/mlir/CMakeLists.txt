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

# The LLVM/MLIR helpers used below (add_mlir_library and friends) compile their
# sources in a separate `obj.<name>` object library and forward the target's
# INCLUDE_DIRECTORIES to it as plain include paths, which drops the SYSTEM
# marking. Marking the third-party headers as system directories for the whole
# subtree survives that, and keeps our warning set (-Werror included) from
# firing inside LLVM, MLIR and spdlog headers.
#
# ${PYTHON_MLIR_BINARY_DIR} holds nothing but TableGen output (Ops.h.inc,
# Passes.h.inc, ...), which is machine-generated and not ours to clean up, so it
# is treated the same way.
include_directories(SYSTEM
                    ${LLVM_INCLUDE_DIRS}
                    ${MLIR_INCLUDE_DIRS}
                    ${spdlog_SOURCE_DIR}/include
                    ${PYTHON_MLIR_BINARY_DIR}
                    ${PYTHON_MLIR_BINARY_DIR}/Dialect)

add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(tools/python-mlir-opt)
add_subdirectory(test)

add_library(python-mlir compile.cpp)
target_link_libraries(python-mlir PRIVATE PythonMLIRDialect TargetPythonBytecode PythonConversionPasses)
python_cpp_link_project_options(python-mlir)
