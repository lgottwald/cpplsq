cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

ExternalProject_Add(libcatch
  DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/catch"
  DOWNLOAD_COMMAND wget https://raw.githubusercontent.com/catchorg/Catch2/Catch1.x/single_include/catch.hpp
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
  CONFIGURE_COMMAND ""
)

FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(cpplsq REQUIRED)
FIND_PACKAGE(libspline)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pedantic-errors -Wall -Wextra -Wno-unused-parameter")
add_definitions(
 ${cpplsq_DEFINITIONS}
)
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${cpplsq_INCLUDE_DIRS}
)

if( libspline_FOUND )
include_directories(
  ${libspline_INCLUDE_DIRS}
)
add_executable( cpplsq_test Main.cpp CholeskyTest.cpp SingleDiffTest.cpp MultiDiffTest.cpp LsqTest.cpp   )
else()
add_executable( cpplsq_test Main.cpp CholeskyTest.cpp MultiDiffTest.cpp LsqTest.cpp   )
endif()
add_dependencies( cpplsq_test libcatch )
target_link_libraries( cpplsq_test ${cpplsq_LIBRARIES} )

enable_testing()
add_test( CpplsqTest cpplsq_test )
