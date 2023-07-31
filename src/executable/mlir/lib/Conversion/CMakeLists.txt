add_subdirectory(PythonToPythonBytecode)

set(PYTHON_CONVERSION_LIBS PythonToPythonBytecode)

add_mlir_library(PythonConversionPasses
    Passes.cpp

    DEPENDS
    PythonConversionPassIncGen

    LINK_LIBS PUBLIC
    ${PYTHON_CONVERSION_LIBS}
)