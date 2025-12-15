include(GoogleTest)

add_executable(python_bytecode_tests
    RegisterAllocation_tests.cpp)
target_link_libraries(python_bytecode_tests PRIVATE
    TargetPythonBytecode
    gtest_main)
gtest_discover_tests(python_bytecode_tests)