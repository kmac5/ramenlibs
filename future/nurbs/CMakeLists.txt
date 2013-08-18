CMAKE_MINIMUM_REQUIRED( VERSION 2.8.0 FATAL_ERROR)

######################################################
# project

PROJECT( nurbs)

######################################################
# cmake stuff

# disable in source builds
IF( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    MESSAGE(FATAL_ERROR "CMake generation for jupiter_core library is not allowed within the source directory!")
ENDIF()

SET( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake/Modules)

IF( NOT CMAKE_BUILD_TYPE)
        SET( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

######################################################
# options

######################################################
# platform compiler flags

IF( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   # Mac OS X
ELSEIF( ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
ELSEIF( ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
   # Windows
ELSE()
    #MESSAGE( FATAL_ERROR "Platform ${CMAKE_SYSTEM_NAME} not supported yet")
ENDIF()

######################################################
# more compiler flags

IF( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX) # OSX and linux
    SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")
ENDIF()

######################################################
# dependencies

# edit if you have other version of boost
SET( Boost_ADDITIONAL_VERSIONS "1.49.0")
SET( Boost_USE_MULTITHREADED ON)
FIND_PACKAGE( Boost REQUIRED)

######################################################
# definitions

######################################################
# sources

ADD_SUBDIRECTORY( nurbs)

######################################################
# tests

ENABLE_TESTING()
ADD_SUBDIRECTORY( test)

