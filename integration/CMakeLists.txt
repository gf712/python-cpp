add_executable(integration-tests_ program.cpp ../src/testing/main.cpp)
target_link_libraries(integration-tests_ PRIVATE python-cpp gtest gtest_main cxxopts project_options project_warnings tsl::ordered_map)
# gtest_discover_tests(integration-tests_)

add_test(
  NAME integration-tests
  COMMAND ${PROJECT_SOURCE_DIR}/integration/run_integration_tests.sh $<TARGET_FILE:python>
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/integration
)

# Ensure the test only runs after the required targets are built
set_tests_properties(integration-tests PROPERTIES
  FIXTURES_REQUIRED "python_built"
)

# Add a setup test that depends on building python
add_test(NAME setup-python COMMAND ${CMAKE_COMMAND} -E echo "Python built")
set_tests_properties(setup-python PROPERTIES
  FIXTURES_SETUP "python_built"
  DEPENDS python  # This may not work in all CMake versions
)
