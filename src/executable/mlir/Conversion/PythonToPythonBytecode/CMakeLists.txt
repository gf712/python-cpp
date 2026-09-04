add_mlir_conversion_library(PythonToPythonBytecode
    PythonToPythonBytecode.cpp
    ArithPatterns.cpp
    AttributeSubscriptPatterns.cpp
    CollectionPatterns.cpp
    ControlFlowPatterns.cpp
    FunctionPatterns.cpp
    ImportPatterns.cpp
    LoadStorePatterns.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/executable/mlir/Conversion/PythonToPythonBytecode

    LINK_LIBS
    MLIRIR
    MLIRPass
    PythonMLIRDialect
    EmitPythonBytecodeDialect
)

target_include_directories(PythonToPythonBytecode SYSTEM PUBLIC
    ${MLIR_INCLUDE_DIRS}
)

target_include_directories(PythonToPythonBytecode PUBLIC
    ${PYTHON_MLIR_SOURCE_DIR}
    ${PYTHON_MLIR_BINARY_DIR}
)

target_link_libraries(PythonToPythonBytecode PRIVATE spdlog)
python_cpp_link_project_options(PythonToPythonBytecode)

# add_mlir_conversion_library also creates an obj.<name> object-library twin, and
# that twin does not inherit the TableGen ordering the named target gets through
# LINK_LIBS. Without this a clean parallel build can start compiling these
# sources before Python/IR/*.inc has been generated and fails with
# "fatal error: Python/IR/PythonTypes.h.inc: No such file or directory".
# It only shows up at some -j values, so it looks intermittent.
add_dependencies(obj.PythonToPythonBytecode
    MLIRPythonOpsIncGen
    MLIREmitPythonBytecodeIncGen
    PythonConversionPassIncGen)
