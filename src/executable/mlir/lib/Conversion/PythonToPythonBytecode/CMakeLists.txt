add_mlir_conversion_library(PythonToPythonBytecode
    PythonToPythonBytecode.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/executable/mlir/include/mlir/Conversion/PythonToPythonBytecode

    LINK_LIBS
    MLIRIR
    MLIRPass
    PythonMLIRDialect
)

target_include_directories(PythonToPythonBytecode PUBLIC 
    ${MLIR_INCLUDE_DIRS} 
    ${PYTHON_MLIR_SOURCE_DIR}
    ${PYTHON_MLIR_BINARY_DIR}
)

target_link_libraries(PythonToPythonBytecode PRIVATE spdlog)