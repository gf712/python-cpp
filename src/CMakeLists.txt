set(AST_SOURCE_FILES # cmake-format: sortable
                     ast/AST.cpp ast/optimizers/ConstantFolding.cpp)

set(BYTECODE_SOURCE_FILES
    # cmake-format: sortable
    executable/bytecode/Bytecode.cpp
    executable/bytecode/codegen/BytecodeGenerator.cpp
    executable/bytecode/instructions/ClearExceptionState.cpp
    executable/bytecode/instructions/FunctionCall.cpp
    executable/bytecode/instructions/FunctionCallWithKeywords.cpp
    executable/bytecode/instructions/GreaterThan.cpp
    executable/bytecode/instructions/InplaceSub.cpp
    executable/bytecode/instructions/IsOp.cpp
    executable/bytecode/instructions/ImportName.cpp
    executable/bytecode/instructions/Instructions.cpp
    executable/bytecode/instructions/JumpForward.cpp
    executable/bytecode/instructions/JumpIfFalseOrPop.cpp
    executable/bytecode/instructions/JumpIfNotExceptionMatch.cpp
    executable/bytecode/instructions/JumpIfTrue.cpp
    executable/bytecode/instructions/JumpIfTrueOrPop.cpp
    executable/bytecode/instructions/LoadAssertionError.cpp
    executable/bytecode/instructions/LoadAttr.cpp
    executable/bytecode/instructions/LoadBuildClass.cpp
    executable/bytecode/instructions/LoadMethod.cpp
    executable/bytecode/instructions/LoadName.cpp
    executable/bytecode/instructions/MethodCall.cpp
    executable/bytecode/instructions/NotEqual.cpp
    executable/bytecode/instructions/RaiseVarargs.cpp
    executable/bytecode/instructions/ReturnValue.cpp
    executable/bytecode/instructions/SetupExceptionHandling.cpp
    executable/bytecode/instructions/StoreAttr.cpp
    executable/bytecode/instructions/StoreName.cpp
    executable/bytecode/instructions/Unary.cpp
    executable/bytecode/instructions/UnpackSequence.cpp)

set(EXECUTABLE_SOURCE_FILES # cmake-format: sortable
                            executable/Program.cpp)

set(LEXER_SOURCE_FILES # cmake-format: sortable
                       lexer/Lexer.cpp)

set(INTERPRETER_SOURCE_FILES
    # cmake-format: sortable
    interpreter/ExecutionFrame.cpp interpreter/Interpreter.cpp interpreter/InterpreterSession.cpp)

set(MEMORY_SOURCE_FILES # cmake-format: sortable
                        memory/GarbageCollector.cpp memory/Heap.cpp)

set(PARSER_SOURCE_FILES # cmake-format: sortable
                        parser/Parser.cpp)

set(RUNTIME_SOURCE_FILES
    # cmake-format: sortable
    runtime/CustomPyObject.cpp
    runtime/modules/BuiltinsModule.cpp
    runtime/modules/SysModule.cpp
    runtime/types/builtin.cpp
    runtime/AssertionError.cpp
    runtime/AttributeError.cpp
    runtime/BaseException.cpp
    runtime/Exception.cpp
    runtime/NameError.cpp
    runtime/PyBoundMethod.cpp
    runtime/PyBool.cpp
    runtime/PyBuiltInMethod.cpp
    runtime/PyBytes.cpp
    runtime/PyDict.cpp
    runtime/PyEllipsis.cpp
    runtime/PyFloat.cpp
    runtime/PyFunction.cpp
    runtime/PyInteger.cpp
    runtime/PyList.cpp
    runtime/PyMethodDescriptor.cpp
    runtime/PyModule.cpp
    runtime/PyNone.cpp
    runtime/PyNumber.cpp
    runtime/PyObject.cpp
    runtime/PyRange.cpp
    runtime/PyString.cpp
    runtime/PyTuple.cpp
    runtime/PyType.cpp
    runtime/PySlotWrapper.cpp
    runtime/PyStaticMethod.cpp
    runtime/StopIteration.cpp
    runtime/TypeError.cpp
    runtime/Value.cpp
    runtime/ValueError.cpp)

set(VM_SOURCE_FILES # cmake-format: sortable
                    vm/VM.cpp)

set(UNITTEST_SOURCES
    # cmake-format: sortable
    ast/optimizers/Optimizers_tests.cpp
    executable/bytecode/Bytecode_tests.cpp
    executable/bytecode/codegen/BytecodeGenerator_tests.cpp
    lexer/Lexer_tests.cpp
    parser/Parser_tests.cpp
    runtime/PyNumber_tests.cpp
    runtime/PyType_tests.cpp)

add_library(
  python-cpp
  ${AST_SOURCE_FILES}
  ${BYTECODE_SOURCE_FILES}
  ${EXECUTABLE_SOURCE_FILES}
  ${LEXER_SOURCE_FILES}
  ${INTERPRETER_SOURCE_FILES}
  ${MEMORY_SOURCE_FILES}
  ${PARSER_SOURCE_FILES}
  ${RUNTIME_SOURCE_FILES}
  ${VM_SOURCE_FILES})

add_executable(unittests ${UNITTEST_SOURCES})

target_link_libraries(python-cpp PUBLIC spdlog_header_only)

target_include_directories(python-cpp PUBLIC .)

# -fno-omit-frame-pointer is needed, otherwise we cannot access the stack base pointer reliably
target_compile_options(
  python-cpp
  PRIVATE -Wall
          -Wextra
          -Werror
          -fno-omit-frame-pointer)

if(STL_SUPPORTS_BIT_CAST)
  target_compile_definitions(python-cpp PUBLIC "STL_SUPPORTS_BIT_CAST")
endif()

target_link_libraries(unittests PRIVATE python-cpp gtest gtest_main)

add_executable(python repl/repl.cpp)
target_link_libraries(python PRIVATE linenoise cxxopts python-cpp)
