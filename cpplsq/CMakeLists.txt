cmake_minimum_required(VERSION 2.8)

find_package(libsimd REQUIRED)

include_directories(
 ${libsimd_INCLUDE_DIRS}
)

add_library( cpplsq STATIC MultiDiff.cpp )

set( ARCH_FLAGS -msse4.1 )

set_target_properties( cpplsq PROPERTIES
    COMPILE_FLAGS 
    "-fPIC -std=c++11 ${ARCH_FLAGS} -pedantic-errors -Wall -Wextra"
     COMPILE_DEFINITIONS
     "PARALLEL=0" )
  
FILE(GLOB header_files "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
INSTALL(FILES ${header_files} DESTINATION include/cpplsq)
INSTALL(TARGETS cpplsq EXPORT cpplsq-targets ARCHIVE DESTINATION lib)

# In my case the folder of includes can be any source folder,
# a practice very extended (in opposition to a single folder
# containing all the headers).
set ( cpplsq_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. )

# Destination of the installed config files (relative path):
set ( CMAKE_CONFIG_DEST "share/cmake/cpplsq" )

include ( CMakePackageConfigHelpers )

# We configure our template. The template is described later.
configure_package_config_file (                    
        "${CMAKE_CURRENT_SOURCE_DIR}/cpplsq-config.cmake.in"
        # Important to write in CMAKE_BINARY_DIR if you want the registry
        # mechanism to work:
        "${CMAKE_BINARY_DIR}/cpplsq-config.cmake"  
        INSTALL_DESTINATION "${CMAKE_CONFIG_DEST}"
        PATH_VARS cpplsq_INCLUDE_DIR libsimd_INCLUDE_DIRS )

# This file is included in our template:
export ( TARGETS cpplsq FILE ${CMAKE_BINARY_DIR}/cpplsq-targets.cmake )

export ( PACKAGE cpplsq )

# We redefine this variable, using this time a relative path:
set ( cpplsq_INCLUDE_DIR "include" )
set ( libsimd_INCLUDE_DIRS "include" )
# We write in the 'export' folder in order not to collide with
# the previous config file:
configure_package_config_file ( 
       "${CMAKE_CURRENT_SOURCE_DIR}/cpplsq-config.cmake.in" 
       "${CMAKE_BINARY_DIR}/export/cpplsq-config.cmake"
       INSTALL_DESTINATION "${CMAKE_CONFIG_DEST}"
       PATH_VARS cpplsq_INCLUDE_DIR libsimd_INCLUDE_DIRS )

install (
      EXPORT cpplsq-targets 
      DESTINATION ${CMAKE_CONFIG_DEST} FILE "cpplsq-targets.cmake" )
install(
  FILES ${CMAKE_BINARY_DIR}/export/cpplsq-config.cmake
  DESTINATION ${CMAKE_CONFIG_DEST}
)
