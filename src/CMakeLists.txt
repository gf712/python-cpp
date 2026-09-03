set(AST_SOURCE_FILES # cmake-format: sortable
                     core.cpp memory/allocate.cpp)

set(BYTECODE_SOURCE_FILES
    # cmake-format: sortable
    executable/bytecode/Bytecode.cpp)

set(EXECUTABLE_SOURCE_FILES # cmake-format: sortable
                            executable/Program.cpp)

set(LEXER_SOURCE_FILES "")

set(LLVM_BACKEND_FILES # cmake-format: sortable
                       executable/llvm/LLVMGenerator.cpp executable/llvm/LLVMProgram.cpp)

set(INTERPRETER_SOURCE_FILES # cmake-format: sortable
                             interpreter/Interpreter.cpp)

set(MEMORY_SOURCE_FILES # cmake-format: sortable
                        memory/GarbageCollector.cpp memory/Heap.cpp)

set(PARSER_SOURCE_FILES "")

# The Python standard library implemented in C++, plus the runtime types that are
# not `py.runtime` partitions yet. These are ordinary consumers - they import the
# module but define nothing belonging to it - so they build in python-cpp.
set(STDLIB_SOURCE_FILES
    # cmake-format: sortable
    runtime/AssertionError.cpp
    runtime/AttributeError.cpp
    runtime/Exception.cpp
    runtime/Import.cpp
    runtime/ImportError.cpp
    runtime/IndexError.cpp
    runtime/KeyError.cpp
    runtime/LookupError.cpp
    runtime/MemoryError.cpp
    runtime/ModuleNotFoundError.cpp
    runtime/NameError.cpp
    runtime/NotImplemented.cpp
    runtime/NotImplementedError.cpp
    runtime/OSError.cpp
    runtime/PyBool.cpp
    runtime/PyBoundMethod.cpp
    runtime/PyBuiltInMethod.cpp
    runtime/PyByteArray.cpp
    runtime/PyBytes.cpp
    runtime/PyClassMethod.cpp
    runtime/PyClassMethodDescriptor.cpp
    runtime/PyComplex.cpp
    runtime/PyEllipsis.cpp
    runtime/PyEnumerate.cpp
    runtime/PyGenericAlias.cpp
    runtime/PyGetSetDescriptor.cpp
    runtime/PyIterator.cpp
    runtime/PyLLVMFunction.cpp
    runtime/PyMap.cpp
    runtime/PyMappingProxy.cpp
    runtime/PyMemberDescriptor.cpp
    runtime/PyMemoryView.cpp
    runtime/PyMethodDescriptor.cpp
    runtime/PyModule.cpp
    runtime/PyNamespace.cpp
    runtime/PyProperty.cpp
    runtime/PyRange.cpp
    runtime/PyReversed.cpp
    runtime/PySet.cpp
    runtime/PySlice.cpp
    runtime/PySlotWrapper.cpp
    runtime/PyStaticMethod.cpp
    runtime/PySuper.cpp
    runtime/PyZip.cpp
    runtime/RuntimeError.cpp
    runtime/SourceManager.cpp
    runtime/StopIteration.cpp
    runtime/SyntaxError.cpp
    runtime/TypeError.cpp
    runtime/UnboundLocalError.cpp
    runtime/ValueError.cpp
    runtime/warnings/DeprecationWarning.cpp
    runtime/warnings/ImportWarning.cpp
    runtime/warnings/PendingDeprecationWarning.cpp
    runtime/warnings/ResourceWarning.cpp)

# Translation units belonging to the `py.runtime` module: the .cpp counterparts of
# the interface partitions listed in python-runtime's FILE_SET. Each defines members
# of a module-owned class, so it must be an implementation unit (`module py.runtime;`)
# compiled in python-runtime - a definition has to share its declaration's module
# attachment, and a plain TU elsewhere is rejected with
# "redeclaring ... in global module conflicts with import".
set(RUNTIME_SOURCE_FILES
    # cmake-format: sortable
    ast/AST.cpp
    ast/ASTArena.cpp
    executable/bytecode/instructions/Instructions.cpp
    executable/bytecode/BytecodeProgram.cpp
    runtime/warnings/Warning.cpp
    runtime/modules/BuiltinsModule.cpp
    runtime/modules/CodecsModule.cpp
    runtime/modules/GcModule.cpp
    runtime/modules/IOModule.cpp
    runtime/modules/ImpModule.cpp
    runtime/modules/SysModule.cpp
    executable/Mangler.cpp
    parser/Parser.cpp
    executable/bytecode/instructions/BinaryOperation.cpp
    executable/bytecode/instructions/BinarySubscript.cpp
    executable/bytecode/instructions/BuildDict.cpp
    executable/bytecode/instructions/BuildList.cpp
    executable/bytecode/instructions/BuildTuple.cpp
    executable/bytecode/instructions/BuildSet.cpp
    executable/bytecode/instructions/BuildSlice.cpp
    executable/bytecode/instructions/BuildString.cpp
    executable/bytecode/instructions/CompareOperation.cpp
    executable/bytecode/instructions/ClearExceptionState.cpp
    executable/bytecode/instructions/ClearTopCleanup.cpp
    executable/bytecode/instructions/DictAdd.cpp
    executable/bytecode/instructions/DictMerge.cpp
    executable/bytecode/instructions/DictUpdate.cpp
    executable/bytecode/instructions/DeleteAttr.cpp
    executable/bytecode/instructions/DeleteDeref.cpp
    executable/bytecode/instructions/DeleteFast.cpp
    executable/bytecode/instructions/DeleteGlobal.cpp
    executable/bytecode/instructions/DeleteName.cpp
    executable/bytecode/instructions/DeleteSubscript.cpp
    executable/bytecode/instructions/ForIter.cpp
    executable/bytecode/instructions/FormatValue.cpp
    executable/bytecode/instructions/FunctionCall.cpp
    executable/bytecode/instructions/FunctionCallEx.cpp
    executable/bytecode/instructions/FunctionCallWithKeywords.cpp
    executable/bytecode/instructions/GetAwaitable.cpp
    executable/bytecode/instructions/GetIter.cpp
    executable/bytecode/instructions/GetYieldFromIter.cpp
    executable/bytecode/instructions/InplaceOp.cpp
    executable/bytecode/instructions/ImportFrom.cpp
    executable/bytecode/instructions/ImportName.cpp
    executable/bytecode/instructions/ImportStar.cpp
    executable/bytecode/instructions/Jump.cpp
    executable/bytecode/instructions/JumpForward.cpp
    executable/bytecode/instructions/JumpIfFalse.cpp
    executable/bytecode/instructions/JumpIfFalseOrPop.cpp
    executable/bytecode/instructions/JumpIfExceptionMatch.cpp
    executable/bytecode/instructions/JumpIfNotExceptionMatch.cpp
    executable/bytecode/instructions/JumpIfTrue.cpp
    executable/bytecode/instructions/JumpIfTrueOrPop.cpp
    executable/bytecode/instructions/LeaveExceptionHandling.cpp
    executable/bytecode/instructions/ListAppend.cpp
    executable/bytecode/instructions/ListExtend.cpp
    executable/bytecode/instructions/ListToTuple.cpp
    executable/bytecode/instructions/LoadAssertionError.cpp
    executable/bytecode/instructions/LoadException.cpp
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
    executable/bytecode/instructions/Pop.cpp
    executable/bytecode/instructions/Push.cpp
    executable/bytecode/instructions/RaiseVarargs.cpp
    executable/bytecode/instructions/ReRaise.cpp
    executable/bytecode/instructions/ReturnValue.cpp
    executable/bytecode/instructions/SetAdd.cpp
    executable/bytecode/instructions/SetUpdate.cpp
    executable/bytecode/instructions/SetupExceptionHandling.cpp
    executable/bytecode/instructions/SetupWith.cpp
    executable/bytecode/instructions/StoreAttr.cpp
    executable/bytecode/instructions/StoreDeref.cpp
    executable/bytecode/instructions/StoreFast.cpp
    executable/bytecode/instructions/StoreGlobal.cpp
    executable/bytecode/instructions/StoreName.cpp
    executable/bytecode/instructions/StoreSubscript.cpp
    executable/bytecode/instructions/ToBool.cpp
    executable/bytecode/instructions/Unary.cpp
    executable/bytecode/instructions/UnpackExpand.cpp
    executable/bytecode/instructions/UnpackSequence.cpp
    executable/bytecode/instructions/WithExceptStart.cpp
    executable/bytecode/instructions/YieldFrom.cpp
    executable/bytecode/instructions/YieldLoad.cpp
    executable/bytecode/instructions/YieldValue.cpp
    runtime/modules/collections/DefaultDict.cpp
    runtime/modules/collections/Deque.cpp
    runtime/modules/itertools/Chain.cpp
    runtime/modules/itertools/Count.cpp
    runtime/modules/itertools/ISlice.cpp
    runtime/modules/itertools/Permutations.cpp
    runtime/modules/itertools/Product.cpp
    runtime/modules/itertools/Repeat.cpp
    runtime/modules/itertools/StarMap.cpp
    runtime/modules/sre/Match.cpp
    runtime/modules/sre/Pattern.cpp
    runtime/modules/weakref/PyCallableProxyType.cpp
    runtime/modules/weakref/PyWeakProxy.cpp
    runtime/modules/weakref/PyWeakRef.cpp
    runtime/modules/MarshalModule.cpp
    runtime/modules/PosixModule.cpp
    runtime/modules/WarningsModule.cpp
    runtime/modules/collections/module.cpp
    runtime/modules/errno/module.cpp
    runtime/modules/itertools/module.cpp
    runtime/modules/math/module.cpp
    runtime/modules/signal/module.cpp
    runtime/modules/sre/module.cpp
    runtime/modules/struct/module.cpp
    runtime/modules/thread/module.cpp
    runtime/modules/time/module.cpp
    runtime/modules/weakref/module.cpp
    executable/bytecode/codegen/BytecodeGenerator.cpp
    executable/bytecode/codegen/VariablesResolver.cpp
    lexer/Lexer.cpp
    interpreter/InterpreterSession.cpp
    runtime/BaseException.cpp
    runtime/GeneratorInterface.cpp
    runtime/PyAsyncGenerator.cpp
    runtime/PyCell.cpp
    runtime/PyCode.cpp
    runtime/PyCoroutine.cpp
    runtime/PyDict.cpp
    runtime/PyFloat.cpp
    runtime/PyFrame.cpp
    runtime/PyFrozenSet.cpp
    runtime/PyFunction.cpp
    runtime/PyGenerator.cpp
    runtime/PyInteger.cpp
    runtime/PyList.cpp
    runtime/PyNone.cpp
    runtime/PyNumber.cpp
    runtime/PyObject.cpp
    runtime/PyString.cpp
    runtime/PyTraceback.cpp
    runtime/PyTuple.cpp
    runtime/PyType.cpp
    runtime/Value.cpp
    runtime/types/builtin.cpp
)

set(VM_SOURCE_FILES # cmake-format: sortable
                    vm/VM.cpp)

set(UNITTEST_SOURCES
    # cmake-format: sortable
    ast/ASTArena_tests.cpp
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
    runtime/SourceManager_tests.cpp
    runtime/warnings/Warnings_tests.cpp
    testing/main.cpp)

set(PYTHON_LIB_PATH ${cpython_SOURCE_DIR}/Lib)
set(PYTHON_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})

configure_file(runtime/modules/paths.hpp.in runtime/modules/paths.hpp)

# ---------------------------------------------------------------------------
# python-runtime: the sole provider of the `py.runtime` module.
# ---------------------------------------------------------------------------
add_library(python-runtime)
set_target_properties(python-runtime PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)

target_sources(python-runtime
  PUBLIC
    FILE_SET CXX_MODULES
    FILES
      executable/Function.cppm
      executable/FunctionBlock.cppm
      executable/Program.cppm
      executable/bytecode/Bytecode.cppm
      executable/bytecode/BytecodeProgram.cppm
      executable/bytecode/codegen/codegen.cppm
      executable/bytecode/codegen/VariablesResolver.cppm
      executable/bytecode/codegen/BytecodeGenerator.cppm
      memory/memory.cppm
      memory/GarbageCollector.cppm
      memory/Heap.cppm
      lexer/Lexer.cppm
      ast/ASTArena.cppm
      ast/AST.cppm
      interpreter/Interpreter.cppm
      interpreter/InterpreterSession.cppm
      runtime/runtime.cppm
      executable/bytecode/instructions/Instructions.cppm
      runtime/Klass.cppm
      runtime/AssertionError.cppm
      runtime/AttributeError.cppm
      runtime/ImportError.cppm
      runtime/IndexError.cppm
      runtime/LookupError.cppm
      runtime/MemoryError.cppm
      runtime/ModuleNotFoundError.cppm
      runtime/NotImplemented.cppm
      runtime/NotImplementedError.cppm
      runtime/OSError.cppm
      runtime/PyArgParser.cppm
      runtime/PyBoundMethod.cppm
      runtime/PyBuiltInMethod.cppm
      runtime/PyByteArray.cppm
      runtime/PyBytes.cppm
      runtime/PyClassMethod.cppm
      runtime/PyClassMethodDescriptor.cppm
      runtime/PyComplex.cppm
      runtime/PyEllipsis.cppm
      runtime/PyEnumerate.cppm
      runtime/PyGenericAlias.cppm
      runtime/PyGetSetDescriptor.cppm
      runtime/PyIterator.cppm
      runtime/PyLLVMFunction.cppm
      runtime/PyMap.cppm
      runtime/PyMappingProxy.cppm
      runtime/PyMemberDescriptor.cppm
      runtime/PyMemoryView.cppm
      runtime/PyMethodDescriptor.cppm
      runtime/PyNamespace.cppm
      runtime/PyProperty.cppm
      runtime/PyRange.cppm
      runtime/PyReversed.cppm
      runtime/PySet.cppm
      runtime/PySlice.cppm
      runtime/PySlotWrapper.cppm
      runtime/PyStaticMethod.cppm
      runtime/PySuper.cppm
      runtime/PyZip.cppm
      runtime/RuntimeError.cppm
      runtime/StopIteration.cppm
      runtime/SyntaxError.cppm
      runtime/TypeError.cppm
      runtime/UnboundLocalError.cppm
      runtime/ValueError.cppm
      runtime/utilities.cppm
      runtime/warnings/DeprecationWarning.cppm
      runtime/warnings/ImportWarning.cppm
      runtime/warnings/PendingDeprecationWarning.cppm
      runtime/warnings/ResourceWarning.cppm
      runtime/warnings/Warning.cppm
      runtime/modules/Modules.cppm
      runtime/BaseException.cppm
      runtime/concepts.cppm
      runtime/Exception.cppm
      runtime/GeneratorInterface.cppm
      runtime/Import.cppm
      runtime/KeyError.cppm
      runtime/NameError.cppm
      runtime/PyAsyncGenerator.cppm
      runtime/PyBool.cppm
      runtime/PyCell.cppm
      runtime/PyCode.cppm
      runtime/PyCoroutine.cppm
      runtime/PyDict.cppm
      runtime/PyFloat.cppm
      runtime/PyFrame.cppm
      runtime/PyFrozenSet.cppm
      runtime/PyFunction.cppm
      runtime/PyGenerator.cppm
      runtime/PyInteger.cppm
      runtime/PyList.cppm
      runtime/PyModule.cppm
      runtime/PyNone.cppm
      runtime/PyNumber.cppm
      runtime/PyObject.cppm
      runtime/PyString.cppm
      runtime/PyTraceback.cppm
      runtime/PyTuple.cppm
      runtime/PyType.cppm
      runtime/Value.cppm
      vm/VM.cppm
      executable/Mangler.cppm
      parser/Parser.cppm
      runtime/types/types.cppm
      runtime/types/builtin.cppm
      runtime/types/api.cppm
  PRIVATE
    ${EXECUTABLE_SOURCE_FILES}
    ${INTERPRETER_SOURCE_FILES}
    ${MEMORY_SOURCE_FILES}
    ${RUNTIME_SOURCE_FILES}
)

target_include_directories(python-runtime
  PUBLIC .
  PRIVATE ${GMP_INCLUDES} ${CMAKE_CURRENT_BINARY_DIR}
)

# Value.cppm exposes mpz_class through py::Number, so GMP is a usage requirement
# of the module interface rather than an implementation detail.
target_link_libraries(python-runtime
  PUBLIC spdlog m ${GMPXX_LIBRARIES} ${GMP_LIBRARIES}
  PRIVATE ICU::uc ICU::data tsl::ordered_map
)

if(STL_SUPPORTS_BIT_CAST)
  target_compile_definitions(python-runtime PUBLIC "STL_SUPPORTS_BIT_CAST")
endif()

python_cpp_link_project_options(python-runtime)

# ---------------------------------------------------------------------------
# python-cpp: everything that is not a `py.runtime` module unit. It consumes the
# module, so it still needs scanning, but it provides no modules of its own.
# ---------------------------------------------------------------------------
add_library(python-cpp)
set_target_properties(python-cpp PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)

target_sources(python-cpp
  PRIVATE
    ${AST_SOURCE_FILES}
    ${BYTECODE_SOURCE_FILES}
    ${LEXER_SOURCE_FILES}
    ${PARSER_SOURCE_FILES}
    ${STDLIB_SOURCE_FILES}
    ${VM_SOURCE_FILES}
)

add_executable(unittests_ ${UNITTEST_SOURCES})
set_target_properties(unittests_ PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)

# PUBLIC so that every consumer of python-cpp (unittests_, python, freeze, and
# the MLIR libraries) inherits an acyclic path to the py.runtime BMIs.
target_link_libraries(python-cpp
  PUBLIC spdlog m python-runtime
  PRIVATE
    ICU::uc
    ICU::data
    ${GMPXX_LIBRARIES}
    ${GMP_LIBRARIES}
    tsl::ordered_map
)

target_include_directories(python-cpp
  PUBLIC .
  PRIVATE ${GMP_INCLUDES} ${CMAKE_CURRENT_BINARY_DIR}
)

python_cpp_link_project_options(python-cpp)

if(STL_SUPPORTS_BIT_CAST)
  target_compile_definitions(python-cpp PUBLIC "STL_SUPPORTS_BIT_CAST")
endif()

# MLIR backend
add_subdirectory(executable/mlir)

target_link_libraries(python-cpp
  PUBLIC
  python-mlir
)

# LLVM backend
# No version is requested here for the same reason as in executable/mlir: LLVM's
# config-version file wants an exact major.minor match. The MLIR subdirectory has
# already validated the major version by this point.
find_package(LLVM CONFIG)
if(ENABLE_LLVM_BACKEND AND NOT LLVM_FOUND)
  message(FATAL_ERROR "Could not find LLVM in the local environment")
elseif(ENABLE_LLVM_BACKEND AND LLVM_FOUND)
  message(STATUS "Configuring LLVM backend")
  add_library(python-cpp-llvm ${LLVM_BACKEND_FILES})

  add_library(llvm-interface INTERFACE)
  target_include_directories(llvm-interface INTERFACE . )
  # include llvm include directories as system paths to silence compiler warnings
  target_include_directories(llvm-interface SYSTEM INTERFACE ${LLVM_INCLUDE_DIRS})
  target_link_libraries(llvm-interface INTERFACE spdlog)
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
  target_link_libraries(llvm-interface INTERFACE ${llvm_libs})
  python_cpp_link_project_options(llvm-interface)
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

target_link_libraries(unittests_ PRIVATE python-cpp gtest gtest_main cxxopts tsl::ordered_map)
gtest_discover_tests(unittests_)

add_executable(python repl/repl.cpp)
set_target_properties(python PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
target_link_libraries(python PRIVATE linenoise cxxopts python-cpp stdc++)

add_executable(freeze utilities/freeze.cpp)
set_target_properties(freeze PROPERTIES CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
target_link_libraries(freeze PRIVATE python-cpp cxxopts)
target_include_directories(freeze SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS})

python_cpp_link_project_options(unittests_ python freeze)
