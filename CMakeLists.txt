cmake_minimum_required(VERSION 3.25)
include(FetchContent)
include(ExternalProject)
include(CheckCXXSourceCompiles)

project(python++)

set(CMAKE_CXX_STANDARD 26)

include(cmake/CPM.cmake)

CPMAddPackage("gh:gabime/spdlog@1.8.5")
CPMAddPackage("gh:google/googletest@1.18.0")
CPMAddPackage("gh:jarro2783/cxxopts@3.3.1")
CPMAddPackage("gh:Tessil/ordered-map@1.2.0")
CPMAddPackage("gh:python/cpython@3.9.25")
CPMAddPackage("gh:antirez/linenoise#2.0")
CPMAddPackage("gh:aminya/project_options@0.41.1")

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
  INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-conversion;-Wno-shadow;-Wno-implicit-fallthrough;-Wno-old-style-cast;-Wno-deprecated-copy;-Wno-missing-field-initializers;-Wno-null-dereference;-Wno-maybe-uninitialized;-Wno-stringop-overflow>
)

target_compile_options(project_options
  # -fno-omit-frame-pointer is needed, otherwise we cannot access the stack base
  # pointer reliably. The garbage collector scans the C++ stack conservatively
  # for roots (see MarkSweepGC::collect_roots), so every target whose frames can
  # be live across an allocation needs it.
  INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
)

# Every first-party target links against project_options and project_warnings
# through this helper so they are all compiled the same way.
include(PythonCppFlags)

# check_cxx_source_compiles links the snippet, so it needs a main(); and
# std::uint64_t needs <cstdint> rather than coming along with <bit>.
check_cxx_source_compiles(
	"#include <bit>
  #include <cstdint>
  constexpr double f64v = 19880124.0;
  constexpr auto u64v = std::bit_cast<std::uint64_t>(f64v);
  int main() { return u64v == 0; }"
	STL_SUPPORTS_BIT_CAST)

find_library(MATH_LIBRARY m)
if(NOT MATH_LIBRARY)
	message(FATAL_ERROR "Could not find math library")
endif()

enable_testing()
add_subdirectory(src)
add_subdirectory(integration)
