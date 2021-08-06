set(AST_SOURCE_FILES ast/AST.cpp 
                     ast/Module.cpp)
set(INTERPRETER_SOURCE_FILES 
                    interpreter/Interpreter.cpp
                    interpreter/ExecutionFrame.cpp)
set(PARSER_SOURCE_FILES parser/Parser.cpp)
set(RUNTIME_SOURCE_FILES runtime/PyDict.cpp 
                         runtime/PyNumber.cpp
                         runtime/PyObject.cpp
                         runtime/PyRange.cpp)

set(VM_SOURCE_FILES bytecode/VM.cpp 
                    bytecode/BytecodeGenerator.cpp
                    bytecode/instructions/FunctionCall.cpp
                    bytecode/instructions/FunctionCallWithKeywords.cpp
                    bytecode/instructions/LoadAttr.cpp
                    bytecode/instructions/LoadMethod.cpp
                    bytecode/instructions/Instructions.cpp
                    bytecode/instructions/MethodCall.cpp
                    bytecode/instructions/ReturnValue.cpp
                    bytecode/instructions/StoreAttr.cpp
                    )

set(UNITTEST_SOURCES lexer/Lexer_tests.cpp parser/Parser_tests.cpp)

add_library(python-cpp ${AST_SOURCE_FILES} ${INTERPRETER_SOURCE_FILES} ${PARSER_SOURCE_FILES} ${RUNTIME_SOURCE_FILES} ${VM_SOURCE_FILES})
add_executable(unittests ${UNITTEST_SOURCES})

target_link_libraries(python-cpp PUBLIC spdlog_header_only)
target_include_directories(python-cpp PUBLIC .)

target_compile_options(python-cpp PRIVATE -Wall -Wextra -Werror)

set(TEST_PROGRAM main.cpp)

add_executable(test ${TEST_PROGRAM})
target_link_libraries(test PRIVATE python-cpp)
target_link_libraries(unittests PRIVATE python-cpp gtest gtest_main)

add_executable(python repl/repl.cpp)
target_link_libraries(python PRIVATE linenoise python-cpp)
