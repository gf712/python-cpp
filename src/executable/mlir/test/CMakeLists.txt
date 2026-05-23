# Lit / FileCheck test harness for the Python and EmitPythonBytecode
# MLIR dialects. Tests live in subdirectories under this one and are
# discovered by lit at run time.

llvm_canonicalize_cmake_booleans(MLIR_ENABLE_BINDINGS_PYTHON)

# add_lit_target looks for lit via LLVM_EXTERNAL_LIT. Default to the
# system-installed lit shipped with the LLVM/MLIR toolchain that
# MLIRConfig.cmake just resolved.
if(NOT LLVM_EXTERNAL_LIT)
    find_program(LLVM_EXTERNAL_LIT NAMES lit lit.py
        HINTS ${LLVM_TOOLS_BINARY_DIR} /usr/lib/llvm-23/bin
    )
endif()

set(MLIR_PYTHON_TEST_DEPENDS
    FileCheck
    python-mlir-opt
)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-mlir-python "Running the python-mlir test suite"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${MLIR_PYTHON_TEST_DEPENDS}
)

set_target_properties(check-mlir-python PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR-PYTHON ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_PYTHON_TEST_DEPENDS})

# Register the check target with ctest so a `ctest --preset release`
# picks it up automatically.
add_test(NAME mlir-python-lit
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-mlir-python
)
