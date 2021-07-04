include(GoogleTest)

add_executable(integration-tests program.cpp)
target_link_libraries(integration-tests PRIVATE python-cpp gtest gtest_main)
gtest_add_tests(TARGET integration-tests)