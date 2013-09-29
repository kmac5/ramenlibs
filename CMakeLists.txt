# Copyright (c) 2013 Esteban Tovagliari

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

CMAKE_MINIMUM_REQUIRED( VERSION 2.8.0 FATAL_ERROR)

######################################################
# project

PROJECT( ramenlibs)

######################################################
# cmake stuff

# disable in source builds
IF( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    MESSAGE(FATAL_ERROR "CMake generation for ramen libraries is not allowed within the source directory!")
ENDIF()

SET( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

IF( NOT CMAKE_BUILD_TYPE)
    SET( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

######################################################
# options

SET( RAMEN_WITH_HALF                ON  CACHE BOOL "Build EXR Half support")
SET( RAMEN_WITH_REGEX               ON  CACHE BOOL "Build regex support")
SET( RAMEN_USE_INTERPROCESS_NAMES   OFF CACHE BOOL "Use interprocess singleton for names")
SET( RAMEN_BUILD_TESTS              ON  CACHE BOOL "Build tests")

# libs compile control
SET( RAMEN_WITH_ARRAYS  ON  CACHE BOOL "Build arrays lib")
SET( RAMEN_WITH_CUDA    ON  CACHE BOOL "Build cuda lib")
SET( RAMEN_WITH_GEO     ON  CACHE BOOL "Build geo lib")
SET( RAMEN_WITH_GEO_IO  ON  CACHE BOOL "Build geo io lib")
SET( RAMEN_WITH_GL      ON  CACHE BOOL "Build gl lib")
SET( RAMEN_WITH_HASH    ON  CACHE BOOL "Build hash lib")
SET( RAMEN_WITH_OS      ON  CACHE BOOL "Build os lib")

SET( RAMEN_BUILD_TOOLS ON CACHE BOOL "Build extra tools")

######################################################
# platform specific

IF( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   # Mac OS X
ELSEIF( ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    ADD_DEFINITIONS( -pthread)
    FIND_PACKAGE( Threads REQUIRED)

    SET( RAMEN_TEST_SYSTEM_LIBS ${CMAKE_THREAD_LIBS_INIT})
ELSEIF( ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
   # Windows
ELSE()
    MESSAGE( FATAL_ERROR "Platform ${CMAKE_SYSTEM_NAME} not supported yet")
ENDIF()

######################################################
# compiler flags

IF( CMAKE_COMPILER_IS_GNUCXX)
    SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fvisibility=hidden -fvisibility-inlines-hidden")
ELSEIF( CMAKE_COMPILER_IS_CLANGXX)
    MESSAGE( FATAL_ERROR "clang is not supported yet")
ELSE()
    MESSAGE( FATAL_ERROR "unknown compiler is not supported yet")
ENDIF()

######################################################
# dependencies

# edit if you have other version of boost
SET( Boost_USE_MULTITHREADED ON)
SET( BOOST_NEEDED_LIBRARIES)
FIND_PACKAGE( Boost COMPONENTS ${BOOST_NEEDED_LIBRARIES} REQUIRED)
ADD_DEFINITIONS( -DBOOST_FILESYSTEM_VERSION=3)
ADD_DEFINITIONS( -DBOOST_FILESYSTEM_NO_DEPRECATED)

IF( RAMEN_WITH_REGEX)
    FIND_PACKAGE( PCRE REQUIRED)
ENDIF()

######################################################
# optional dependencies

IF( RAMEN_WITH_HALF)
    FIND_PACKAGE( Half REQUIRED)
    ADD_DEFINITIONS( -DRAMEN_WITH_HALF)
    INCLUDE_DIRECTORIES( ${HALF_INCLUDE_DIRS})
ENDIF()

IF( RAMEN_WITH_GL)
    FIND_PACKAGE( OpenGL REQUIRED)
    FIND_PACKAGE( GLEW REQUIRED)
ENDIF()

IF( RAMEN_WITH_CUDA)
    FIND_PACKAGE( CUDA REQUIRED)
    FIND_PACKAGE( Thrust REQUIRED)

    # Remove libcuda from CUDA_LIBRARIES ( as it's part of the nvidia driver.)
    SET( CUDA_LIBRARIES ${CUDA_CUDART_LIBRARIES})
ENDIF()

######################################################
# definitions

IF( RAMEN_USE_INTERPROCESS_NAMES)
    ADD_DEFINITIONS( -DRAMEN_USE_INTERPROCESS_NAMES)
ENDIF()

######################################################
# includes

INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR}
                     ramen/config
                     )

######################################################
# libraries

ADD_SUBDIRECTORY( ramen/algorithm)
ADD_SUBDIRECTORY( ramen/bezier)
ADD_SUBDIRECTORY( ramen/color)
ADD_SUBDIRECTORY( ramen/containers)
ADD_SUBDIRECTORY( ramen/core)
ADD_SUBDIRECTORY( ramen/functional)
ADD_SUBDIRECTORY( ramen/iterators)
ADD_SUBDIRECTORY( ramen/math)
ADD_SUBDIRECTORY( ramen/nurbs)
ADD_SUBDIRECTORY( ramen/string_algo)

IF( RAMEN_WITH_ARRAYS)
    ADD_SUBDIRECTORY( ramen/arrays)
ENDIF()

IF( RAMEN_WITH_CUDA)
    ADD_SUBDIRECTORY( ramen/cuda)
ENDIF()

IF( RAMEN_WITH_GEO)
    ADD_SUBDIRECTORY( ramen/geo)
ENDIF()

IF( RAMEN_WITH_GL)
    ADD_SUBDIRECTORY( ramen/gl)
ENDIF()

IF( RAMEN_WITH_HASH)
    ADD_SUBDIRECTORY( ramen/hash)
ENDIF()

IF( RAMEN_WITH_OS)
    ADD_SUBDIRECTORY( ramen/os)
ENDIF()

######################################################
# tests

IF( RAMEN_BUILD_TESTS)
    INCLUDE_DIRECTORIES(  google_testing_libs/google
                          google_testing_libs/google/gtest
                          google_testing_libs/google/gmock
                          )

    ADD_LIBRARY( gmock STATIC   google_testing_libs/google/gtest/src/gtest-all.cc
                                google_testing_libs/google/gmock/src/gmock-all.cc
                                )
    ENABLE_TESTING()
ENDIF()
