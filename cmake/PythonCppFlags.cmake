# Helper for giving every first-party target the same compiler flags.
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
      else()
        target_link_libraries(${name} PRIVATE project_options project_warnings)
      endif()
    endforeach()
  endforeach()
endfunction()
