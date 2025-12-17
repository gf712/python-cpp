add_mlir_conversion_library(PythonToPythonBytecode
    PythonToPythonBytecode.cpp

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