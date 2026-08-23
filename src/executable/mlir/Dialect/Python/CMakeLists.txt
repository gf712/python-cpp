add_subdirectory(IR)

# enable exception handling for now..
set(LLVM_REQUIRES_EH ON)

add_mlir_library(PythonMLIRDialect
                 MLIRGenerator.cpp
                 IR/Ops.cpp

                 DEPENDS
                 MLIRPythonOpsIncGen

                 LINK_LIBS PUBLIC
                 MLIRFuncDialect
                 MLIRSCFDialect
                 MLIRIR
                 MLIRPass
                 MLIRSupport
                 MLIRTransforms
                 MLIRTransformUtils
                )

# The MLIR headers come from the subtree-wide include_directories(SYSTEM ...)
# set up in src/executable/mlir/CMakeLists.txt.
target_include_directories(PythonMLIRDialect PUBLIC
                            ${PROJECT_SOURCE_DIR}/src
                            ${PROJECT_SOURCE_DIR}/src/executable/mlir/Dialect
                            ${CMAKE_BINARY_DIR}/src/executable/mlir/Dialect)
target_link_libraries(PythonMLIRDialect
                      PRIVATE
                        ${GMPXX_LIBRARIES}
                        ${GMP_LIBRARIES}
                      PUBLIC
                        spdlog)
python_cpp_link_project_options(PythonMLIRDialect)