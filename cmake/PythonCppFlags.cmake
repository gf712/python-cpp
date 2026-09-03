# Helper for giving every first-party target the same compiler flags and the
# same C++ module configuration.
#
# The flags come from the external `project_options` package (added with CPM in
# the top-level CMakeLists.txt), which exposes them as two INTERFACE targets:
#   * project_options  - sanitizers, hardening, linker and optimisation flags
#   * project_warnings - the warning set plus -Werror
# The top-level CMakeLists.txt adjusts both to taste; everything else just links
# against them through `python_cpp_link_project_options()` below. Third-party
# code pulled in by CPM (spdlog, googletest, linenoise, ...) is deliberately
# left alone.
#
# The helper exists because of the LLVM/MLIR target helpers
# (add_mlir_library, add_mlir_conversion_library, add_mlir_translation_library,
# ...): they compile their sources in a separate `obj.<name>` object library
# rather than in `<name>` itself, and only forward include directories to it -
# not the usage requirements of libraries linked afterwards. Linking the flags
# to `<name>` alone would therefore silently compile nothing with them, so this
# always covers the `obj.<name>` twin as well.
#
# The same `obj.<name>` split applies to C++ module settings, and there it is
# easier to miss: `CXX_SCAN_FOR_MODULES` set on `<name>` does not reach the
# object library that actually compiles the sources, so those sources are built
# by CMake's "unscanned" rule with no `-fmodule-mapper`. A TU that imports
# `py.runtime` then fails with either "'import' does not name a type" or, worse,
# a fallback lookup in `gcm.cache/`. Setting the properties on both twins is
# what makes `import py.runtime;` work inside the MLIR layer.
#
# Note: do NOT add `-fmodules` here. CMake supplies `-fmodules-ts` together with
# `-fmodule-mapper=` on its scanned compile rules; adding the flag by hand also
# applies it to unscanned targets, which turns a clear diagnostic into a
# confusing module-not-found error.

include_guard(GLOBAL)

function(python_cpp_link_project_options)
  foreach(target ${ARGN})
    foreach(name ${target} obj.${target})
      if(NOT TARGET ${name})
        continue()
      endif()
      get_target_property(type ${name} TYPE)
      if(type STREQUAL "INTERFACE_LIBRARY")
        target_link_libraries(${name} INTERFACE project_options project_warnings)
        continue()
      endif()

      target_link_libraries(${name} PRIVATE project_options project_warnings)

      # Everything first-party either provides or consumes `py.runtime`, so
      # scan it all. Scanning costs ~0.16s per TU (a preprocess-only pass) and
      # removes a whole class of "this target cannot see the module" failures.
      set_target_properties(${name} PROPERTIES CXX_SCAN_FOR_MODULES ON
                                               CXX_MODULE_STD ON)

      # Module imports resolve through link dependencies, so every consumer
      # needs a path to python-runtime - the sole provider of `py.runtime`.
      # It is linked directly rather than via python-cpp because python-cpp and
      # python-mlir are mutually dependent: a module provider reached only
      # through a link cycle cannot be ordered before its consumers, and they
      # compile with an empty module map. python-runtime itself sits below that
      # cycle, so linking it here is always acyclic.
      if(TARGET python-runtime AND NOT ${target} STREQUAL "python-runtime")
        target_link_libraries(${name} PRIVATE python-runtime)
      endif()
    endforeach()
  endforeach()
endfunction()
