# This file was copied from the Eigen project and adapted
# https://gitlab.com/libeigen/eigen/-/blob/400bc5cd5b351a72263737194a5738e6f638d5e8/cmake/FindGMP.cmake

# Try to find the GNU Multiple Precision Arithmetic Library (GMP)
# See http://gmplib.org/

if (GMP_INCLUDES AND GMP_LIBRARIES)
  set(GMP_FIND_QUIETLY TRUE)
endif ()

find_path(GMP_INCLUDES
  NAMES
  gmp.h
  PATHS
  $ENV{GMPDIR}
  ${INCLUDE_INSTALL_DIR}
)

# only look for static libraries
set(_pythoncpp_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES .a)

find_library(GMP_LIBRARIES gmp PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})
find_library(GMPXX_LIBRARIES gmpxx PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

set(${CMAKE_FIND_LIBRARY_SUFFIXES} _pythoncpp_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP DEFAULT_MSG
                                  GMP_INCLUDES GMP_LIBRARIES GMPXX_LIBRARIES)
mark_as_advanced(GMP_INCLUDES GMP_LIBRARIES GMPXX_LIBRARIES)
