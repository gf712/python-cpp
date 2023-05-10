cmake_minimum_required(VERSION 3.16)
include(FetchContent)
include(ExternalProject)
include(CheckCXXSourceCompiles)

project(python++)

set(CMAKE_CXX_STANDARD 20)

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

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(GMP REQUIRED)

find_package(ICU REQUIRED COMPONENTS uc data)
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
include(cmake/project_options.cmake)

option(ENABLE_CACHE "Enable build cache" ON)
option(ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
option(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR "Enable undefined behavior sanitizer" OFF)
option(ENABLE_LLVM_BACKEND "Enable LLVM as a Python execution backend" OFF)

if (${ENABLE_CACHE})
	set(ENABLE_CACHE "ENABLE_CACHE")
endif()

if (${ENABLE_SANITIZER_ADDRESS})
	set(ENABLE_SANITIZER_ADDRESS "ENABLE_SANITIZER_ADDRESS")
endif()

if (${ENABLE_SANITIZER_UNDEFINED_BEHAVIOR})
	set(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR "ENABLE_SANITIZER_UNDEFINED_BEHAVIOR")
endif()

project_options(
	WARNINGS_AS_ERRORS
	# ENABLE_COVERAGE
	# ENABLE_CPPCHECK
	# ENABLE_CLANG_TIDY
	# ENABLE_VS_ANALYSIS
	# ENABLE_INCLUDE_WHAT_YOU_USE
	${ENABLE_CACHE}
	# ENABLE_PCH
	# ENABLE_CONAN
	# ENABLE_VCPKG
	# ENABLE_DOXYGEN
	# ENABLE_INTERPROCEDURAL_OPTIMIZATION
	# ENABLE_NATIVE_OPTIMIZATION
	# ENABLE_USER_LINKER
	# ENABLE_BUILD_WITH_TIME_TRACE
	# ENABLE_UNITY
	${ENABLE_SANITIZER_ADDRESS}
	# ENABLE_SANITIZER_LEAK
	${ENABLE_SANITIZER_UNDEFINED_BEHAVIOR}
	# ENABLE_SANITIZER_THREAD
	# ENABLE_SANITIZER_MEMORY
)

# TODO: Address all the warning below. This should be only temporary...
target_compile_options(project_warnings
  INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-conversion;-Wno-shorten-64-to-32;-Wno-shadow;-Wno-implicit-fallthrough;-Wno-implicit-int-conversion;-Wno-old-style-cast;-Wno-gnu-zero-variadic-macro-arguments;-Wno-implicit-int-float-conversion;-Wno-deprecated-copy>
)

check_cxx_source_compiles(
	"#include <bit>
  constexpr double f64v = 19880124.0;
  constexpr auto u64v = std::bit_cast<std::uint64_t>(f64v);"
	STL_SUPPORTS_BIT_CAST)

add_subdirectory(src)
add_subdirectory(integration)

add_custom_target(test)
add_dependencies(test run-unittests integration-tests)
