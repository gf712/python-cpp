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
                )

target_include_directories(PythonMLIRDialect PUBLIC 
                            ${PROJECT_SOURCE_DIR}/src
                            ${MLIR_INCLUDE_DIRS}
                            ${PROJECT_SOURCE_DIR}/src/executable/mlir/Dialect
                            ${CMAKE_BINARY_DIR}/src/executable/mlir/Dialect)
target_link_libraries(PythonMLIRDialect PRIVATE project_options spdlog)