set(LLVM_TARGET_DEFINITIONS Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls EXTRA_INCLUDES ${MLIR_INCLUDE_DIRS})
add_public_tablegen_target(PythonConversionPassIncGen)

add_subdirectory(PythonToPythonBytecode)

set(PYTHON_CONVERSION_LIBS PythonToPythonBytecode)

add_mlir_library(PythonConversionPasses
    Passes.cpp

    DEPENDS
    PythonConversionPassIncGen

    LINK_LIBS PUBLIC
    ${PYTHON_CONVERSION_LIBS}
)