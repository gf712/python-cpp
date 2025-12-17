cmake_minimum_required(VERSION 3.25)
include(FetchContent)
include(ExternalProject)
include(CheckCXXSourceCompiles)

project(python++)

set(CMAKE_CXX_STANDARD 20)

include(cmake/CPM.cmake)

CPMAddPackage("gh:gabime/spdlog@1.16.0")
CPMAddPackage("gh:google/googletest@1.17.0")
CPMAddPackage("gh:jarro2783/cxxopts@3.0.0")
CPMAddPackage("gh:Tessil/ordered-map@1.2.0")
CPMAddPackage("gh:python/cpython@3.9.25")
CPMAddPackage("gh:antirez/linenoise#1.0")
CPMAddPackage("gh:aminya/project_options@0.41.0")

if(linenoise_ADDED)
  add_library(linenoise ${linenoise_SOURCE_DIR}/linenoise.c)
  target_include_directories(linenoise SYSTEM PUBLIC ${linenoise_SOURCE_DIR})
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(GMP REQUIRED)
find_package(ICU REQUIRED COMPONENTS uc data)

add_library(GTest::GTest ALIAS gtest)
add_library(GTest::GMock ALIAS gmock)
add_library(GTest::Main ALIAS gtest_main)

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
  INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-conversion;-Wno-shadow;-Wno-implicit-fallthrough;-Wno-old-style-cast;-Wno-deprecated-copy;-Wno-missing-field-initializers;-Wno-null-dereference;-Wno-maybe-uninitialized>
)

check_cxx_source_compiles(
	"#include <bit>
  constexpr double f64v = 19880124.0;
  constexpr auto u64v = std::bit_cast<std::uint64_t>(f64v);"
	STL_SUPPORTS_BIT_CAST)

find_library(MATH_LIBRARY m)
if(NOT MATH_LIBRARY)
	message(FATAL_ERROR "Could not find math library")
endif()

enable_testing()
add_subdirectory(src)
add_subdirectory(integration)
