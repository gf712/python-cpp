include(GoogleTest)

add_executable(integration-tests_ program.cpp ../src/testing/main.cpp)
target_link_libraries(integration-tests_ PRIVATE python-cpp gtest gtest_main)
gtest_add_tests(TARGET integration-tests_)

# cmake-format: off
add_custom_target(
  integration-tests
  COMMAND $<TARGET_FILE:integration-tests_>
          && echo "------------------------"
          && echo "Running python scripts:"
          && echo "------------------------"
          && echo ""
          && $<TARGET_FILE:python> ${PROJECT_SOURCE_DIR}/integration/fibonacci/main.py
          && $<TARGET_FILE:python> ${PROJECT_SOURCE_DIR}/integration/mandelbrot/mandelbrot.py
          && ${PROJECT_SOURCE_DIR}/integration/run_python_tests.sh $<TARGET_FILE:python>
          && echo ""
          && echo "------------------------"
          && echo "Testing LLVM backend:"
          && echo "------------------------"
          && echo ""
          && ${PROJECT_SOURCE_DIR}/integration/run_llvm_python_tests.sh $<TARGET_FILE:python>
  DEPENDS integration-tests_ python)
# cmake-format: on
