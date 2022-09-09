cmake_minimum_required(VERSION 3.16)
include(FetchContent)
include(ExternalProject)

project(python++)

set(CMAKE_CXX_STANDARD 20)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
	# Support Unix Makefiles and Ninja
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

FetchContent_Declare(
	spdlog
	GIT_REPOSITORY https://github.com/gabime/spdlog.git
	GIT_TAG v1.8.5)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
	gtest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG release-1.11.0)
FetchContent_MakeAvailable(gtest)

FetchContent_Declare(
	cxxopts
	GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
	GIT_TAG v3.0.0)
FetchContent_MakeAvailable(cxxopts)

FetchContent_Declare(
	cpython
	GIT_REPOSITORY https://github.com/python/cpython.git
	GIT_TAG v3.9.13)
FetchContent_MakeAvailable(cpython)

find_package(ICU REQUIRED COMPONENTS uc)
# if (NOT ${ICU_FOUND})
# 	ExternalProject_Add(icu
# 		GIT_REPOSITORY    https://github.com/unicode-org/icu.git
# 		GIT_TAG           release-71-1
# 		INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/external
# 		CONFIGURE_COMMAND "./configure"
# 		BUILD_COMMAND ${CMAKE_COMMAND} -E env
# 			CC=${CMAKE_C_COMPILER}
# 			CXX=${CMAKE_CXX_COMPILER}
# 			make
# 		BUILD_IN_SOURCE TRUE
# 		INSTALL_COMMAND make DESTDIR=${CMAKE_CURRENT_BINARY_DIR}/external install
# 	)
# 	add_library(icu::)
# endif()

add_library(GTest::GTest ALIAS gtest)
add_library(GTest::GMock ALIAS gmock)
add_library(GTest::Main ALIAS gtest_main)
include(cmake/linenoise.cmake)

include(CheckCXXSourceCompiles)

check_cxx_source_compiles(
	"#include <bit>
  constexpr double f64v = 19880124.0;
  constexpr auto u64v = std::bit_cast<std::uint64_t>(f64v);"
	STL_SUPPORTS_BIT_CAST)

add_subdirectory(src)
add_subdirectory(integration)

add_custom_target(test)
add_dependencies(test run-unittests integration-tests)