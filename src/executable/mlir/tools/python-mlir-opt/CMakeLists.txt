add_llvm_executable(python-mlir-opt
    python-mlir-opt.cpp
)

llvm_update_compile_flags(python-mlir-opt)

target_link_libraries(python-mlir-opt PRIVATE
    PythonMLIRDialect
    EmitPythonBytecodeDialect
    PythonConversionPasses
    PythonToPythonBytecode
    TargetPythonBytecode
    MLIRIR
    MLIROptLib
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRFuncDialect
    MLIRControlFlowDialect
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
)

target_include_directories(python-mlir-opt PRIVATE
    ${PYTHON_MLIR_SOURCE_DIR}
    ${PYTHON_MLIR_BINARY_DIR}
)
