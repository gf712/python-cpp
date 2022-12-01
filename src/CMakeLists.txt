set(AST_SOURCE_FILES # cmake-format: sortable
                     ast/AST.cpp ast/optimizers/ConstantFolding.cpp)

set(BYTECODE_SOURCE_FILES
    # cmake-format: sortable
    executable/bytecode/Bytecode.cpp
    executable/bytecode/BytecodeProgram.cpp
    executable/bytecode/codegen/BytecodeGenerator.cpp
    executable/bytecode/codegen/VariablesResolver.cpp
    executable/bytecode/instructions/BinaryOperation.cpp
    executable/bytecode/instructions/BinarySubscript.cpp
    executable/bytecode/instructions/BuildDict.cpp
    executable/bytecode/instructions/BuildList.cpp
    executable/bytecode/instructions/BuildTuple.cpp
    executable/bytecode/instructions/BuildSet.cpp
    executable/bytecode/instructions/BuildSlice.cpp
    executable/bytecode/instructions/CompareOperation.cpp
    executable/bytecode/instructions/ClearExceptionState.cpp
    executable/bytecode/instructions/ClearTopCleanup.cpp
    executable/bytecode/instructions/DictMerge.cpp
    executable/bytecode/instructions/DictUpdate.cpp
    executable/bytecode/instructions/DeleteName.cpp
    executable/bytecode/instructions/DeleteSubscript.cpp
    executable/bytecode/instructions/ForIter.cpp
    executable/bytecode/instructions/FunctionCall.cpp
    executable/bytecode/instructions/FunctionCallEx.cpp
    executable/bytecode/instructions/FunctionCallWithKeywords.cpp
    executable/bytecode/instructions/GetIter.cpp
    executable/bytecode/instructions/InplaceOp.cpp
    executable/bytecode/instructions/Instructions.cpp
    executable/bytecode/instructions/ImportFrom.cpp
    executable/bytecode/instructions/ImportName.cpp
    executable/bytecode/instructions/Jump.cpp
    executable/bytecode/instructions/JumpForward.cpp
    executable/bytecode/instructions/JumpIfFalse.cpp
    executable/bytecode/instructions/JumpIfFalseOrPop.cpp
    executable/bytecode/instructions/JumpIfNotExceptionMatch.cpp
    executable/bytecode/instructions/JumpIfTrue.cpp
    executable/bytecode/instructions/JumpIfTrueOrPop.cpp
    executable/bytecode/instructions/LeaveExceptionHandling.cpp
    executable/bytecode/instructions/ListAppend.cpp
    executable/bytecode/instructions/ListExtend.cpp
    executable/bytecode/instructions/ListToTuple.cpp
    executable/bytecode/instructions/LoadAssertionError.cpp
    executable/bytecode/instructions/LoadAttr.cpp
    executable/bytecode/instructions/LoadBuildClass.cpp
    executable/bytecode/instructions/LoadClosure.cpp
    executable/bytecode/instructions/LoadConst.cpp
    executable/bytecode/instructions/LoadDeref.cpp
    executable/bytecode/instructions/LoadFast.cpp
    executable/bytecode/instructions/LoadGlobal.cpp
    executable/bytecode/instructions/LoadMethod.cpp
    executable/bytecode/instructions/LoadName.cpp
    executable/bytecode/instructions/MakeFunction.cpp
    executable/bytecode/instructions/MethodCall.cpp
    executable/bytecode/instructions/Move.cpp
    executable/bytecode/instructions/RaiseVarargs.cpp
    executable/bytecode/instructions/ReRaise.cpp
    executable/bytecode/instructions/ReturnValue.cpp
    executable/bytecode/instructions/SetAdd.cpp
    executable/bytecode/instructions/SetupExceptionHandling.cpp
    executable/bytecode/instructions/SetupWith.cpp
    executable/bytecode/instructions/StoreAttr.cpp
    executable/bytecode/instructions/StoreDeref.cpp
    executable/bytecode/instructions/StoreFast.cpp
    executable/bytecode/instructions/StoreGlobal.cpp
    executable/bytecode/instructions/StoreName.cpp
    executable/bytecode/instructions/StoreSubscript.cpp
    executable/bytecode/instructions/Unary.cpp
    executable/bytecode/instructions/UnpackSequence.cpp
    executable/bytecode/instructions/WithExceptStart.cpp
    executable/bytecode/instructions/YieldValue.cpp
    executable/Mangler.cpp)

set(EXECUTABLE_SOURCE_FILES # cmake-format: sortable
                            executable/Program.cpp)

set(LEXER_SOURCE_FILES # cmake-format: sortable
                       lexer/Lexer.cpp)

set(LLVM_BACKEND_FILES # cmake-format: sortable
                       executable/llvm/LLVMGenerator.cpp executable/llvm/LLVMProgram.cpp)

set(INTERPRETER_SOURCE_FILES
    # cmake-format: sortable
    interpreter/Interpreter.cpp interpreter/InterpreterSession.cpp)

set(MEMORY_SOURCE_FILES # cmake-format: sortable
                        memory/GarbageCollector.cpp memory/Heap.cpp)

set(PARSER_SOURCE_FILES # cmake-format: sortable
                        parser/Parser.cpp)

set(RUNTIME_SOURCE_FILES
    # cmake-format: sortable
    runtime/CustomPyObject.cpp
    runtime/modules/BuiltinsModule.cpp
    runtime/modules/ImpModule.cpp
    runtime/modules/IOModule.cpp
    runtime/modules/MarshalModule.cpp
    runtime/modules/PosixModule.cpp
    runtime/modules/SysModule.cpp
    runtime/modules/ThreadModule.cpp
    runtime/modules/WarningsModule.cpp
    runtime/modules/WeakRefModule.cpp
    runtime/types/builtin.cpp
    runtime/AssertionError.cpp
    runtime/AttributeError.cpp
    runtime/BaseException.cpp
    runtime/Exception.cpp
    runtime/GeneratorInterface.cpp
    runtime/IndexError.cpp
    runtime/Import.cpp
    runtime/ImportError.cpp
    runtime/KeyError.cpp
    runtime/LookupError.cpp
    runtime/MemoryError.cpp
    runtime/ModuleNotFoundError.cpp
    runtime/NameError.cpp
    runtime/NotImplemented.cpp
    runtime/NotImplementedError.cpp
    runtime/OSError.cpp
    runtime/PyAsyncGenerator.cpp
    runtime/PyBoundMethod.cpp
    runtime/PyBool.cpp
    runtime/PyBuiltInMethod.cpp
    runtime/PyBytes.cpp
    runtime/PyByteArray.cpp
    runtime/PyCell.cpp
    runtime/PyClassMethod.cpp
    runtime/PyClassMethodDescriptor.cpp
    runtime/PyCode.cpp
    runtime/PyCoroutine.cpp
    runtime/PyDict.cpp
    runtime/PyEllipsis.cpp
    runtime/PyFloat.cpp
    runtime/PyFrame.cpp
    runtime/PyFrozenSet.cpp
    runtime/PyFunction.cpp
    runtime/PyGenerator.cpp
    runtime/PyGenericAlias.cpp
    runtime/PyGetSetDescriptor.cpp
    runtime/PyInteger.cpp
    runtime/PyList.cpp
    runtime/PyLLVMFunction.cpp
    runtime/PyMemberDescriptor.cpp
    runtime/PyMethodDescriptor.cpp
    runtime/PyMemberDescriptor.cpp
    runtime/PyModule.cpp
    runtime/PyNamespace.cpp
    runtime/PyNone.cpp
    runtime/PyNumber.cpp
    runtime/PyObject.cpp
    runtime/PyProperty.cpp
    runtime/PyRange.cpp
    runtime/PyReversed.cpp
    runtime/PySet.cpp
    runtime/PySlice.cpp
    runtime/PySlotWrapper.cpp
    runtime/PyString.cpp
    runtime/PySuper.cpp
    runtime/PyStaticMethod.cpp
    runtime/PyTraceback.cpp
    runtime/PyTuple.cpp
    runtime/PyType.cpp
    runtime/PyWeakRef.cpp
    runtime/PyZip.cpp
    runtime/RuntimeError.cpp
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
    executable/bytecode/BytecodeProgram_tests.cpp
    executable/bytecode/codegen/BytecodeGenerator_tests.cpp
    executable/bytecode/codegen/VariablesResolver_tests.cpp
    lexer/Lexer_tests.cpp
    memory/GarbageCollector_tests.cpp
    memory/Heap_tests.cpp
    parser/Parser_tests.cpp
    runtime/PyDict_tests.cpp
    runtime/PyNumber_tests.cpp
    runtime/PyString_tests.cpp
    runtime/PyType_tests.cpp
    testing/main.cpp)

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

add_executable(unittests_ ${UNITTEST_SOURCES})

target_link_libraries(python-cpp
  PUBLIC spdlog_header_only
  PRIVATE project_options project_warnings ICU::uc
)

target_include_directories(python-cpp PUBLIC .)

target_compile_options(
  python-cpp
  PRIVATE -Wall
  -Wextra
  -Werror
  # -fno-omit-frame-pointer is needed, otherwise we cannot access the stack base pointer reliably
  -fno-omit-frame-pointer)

if(STL_SUPPORTS_BIT_CAST)
  target_compile_definitions(python-cpp PUBLIC "STL_SUPPORTS_BIT_CAST")
endif()

find_package(LLVM CONFIG 10.0)
if(ENABLE_LLVM_BACKEND AND NOT LLVM_FOUND)
  message(FATAL_ERROR "Could not find LLVM in the local environment")
elseif(ENABLE_LLVM_BACKEND AND LLVM_FOUND)
  message(STATUS "Configuring LLVM backend")
  add_library(python-cpp-llvm ${LLVM_BACKEND_FILES})

  target_compile_options(
    python-cpp-llvm
    PRIVATE -Wall
            -Wextra
            -Werror
            -Wno-unused-parameter
            -fno-omit-frame-pointer)

  add_library(llvm-interface INTERFACE)
  target_include_directories(llvm-interface INTERFACE . )
  # include llvm include directories as system paths to silence compiler warnings
  target_include_directories(llvm-interface SYSTEM INTERFACE ${LLVM_INCLUDE_DIRS})
  target_link_libraries(llvm-interface INTERFACE spdlog_header_only)
  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  target_compile_definitions(llvm-interface INTERFACE ${LLVM_DEFINITIONS_LIST})

  if(STL_SUPPORTS_BIT_CAST)
    target_compile_definitions(llvm-interface INTERFACE "STL_SUPPORTS_BIT_CAST")
  endif()
  llvm_map_components_to_libnames(
    llvm_libs
    core
    orcjit
    x86asmparser
    x86codegen)
  target_link_libraries(llvm-interface INTERFACE ${llvm_libs} project_options project_warnings)
  # TODO: not all versions of llvm are ready for C++20, figure out when to use this
  set_property(TARGET python-cpp-llvm PROPERTY CXX_STANDARD 17)

  add_library(python-cpp-llvm-tests OBJECT "executable/llvm/LLVMGenerator_tests.cpp")
  add_library(python-cpp-llvm-glue OBJECT "executable/llvm/LLVMPyUtils.cpp" "executable/llvm/LLVMFunction.cpp")

  target_link_libraries(python-cpp-llvm-glue PRIVATE llvm-interface)
  target_link_libraries(python-cpp-llvm PRIVATE llvm-interface)
  target_sources(python-cpp-llvm PRIVATE $<TARGET_OBJECTS:python-cpp-llvm-glue>)
  target_link_libraries(python-cpp-llvm-tests PRIVATE python-cpp-llvm gtest llvm-interface)
  target_link_libraries(python-cpp PRIVATE python-cpp-llvm)
  target_compile_definitions(python-cpp PUBLIC USE_LLVM)

  target_sources(unittests_ PRIVATE $<TARGET_OBJECTS:python-cpp-llvm-tests>)
  target_link_libraries(unittests_ PRIVATE python-cpp-llvm)
endif()

target_link_libraries(unittests_ PRIVATE python-cpp gtest gtest_main cxxopts project_options project_warnings)
set_target_properties(unittests_ PROPERTIES OUTPUT_NAME "unittests")
add_custom_target(run-unittests COMMAND $<TARGET_FILE:unittests_> DEPENDS unittests_)

add_executable(python repl/repl.cpp)
target_link_libraries(python PRIVATE linenoise cxxopts python-cpp project_options project_warnings)

add_executable(freeze utilities/freeze.cpp)
target_link_libraries(freeze PRIVATE python-cpp cxxopts  project_options project_warnings)
