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

FIND_PACKAGE( Boost COMPONENTS ${BOOST_NEEDED_LIBRARIES} filesystem system REQUIRED)

INCLUDE_DIRECTORIES(    ../../
                        ${Boost_INCLUDE_DIR}
                        ../../../core
                        )

# lib
FILE( GLOB_RECURSE RAMEN_OS_HEADERS ./*.hpp)
SET( RAMEN_OS_SOURCES system_info.cpp
                      )

SET( RAMEN_OS_LIBS  ramen_core
                    ${RAMEN_CORE_LIBS}
                    ${Boost_LIBRARIES}
                    )

IF( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   # Mac OS X
    SET( RAMEN_OS_LIBS ${RAMEN_OS_LIBS})
ELSEIF( ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    # Linux
    SET( RAMEN_OS_LIBS ${RAMEN_OS_LIBS})
ELSEIF( ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
   # Windows
    SET( RAMEN_OS_LIBS ${RAMEN_OS_LIBS})
ELSE()
    MESSAGE( FATAL_ERROR "Platform ${CMAKE_SYSTEM_NAME} not supported yet")
ENDIF()

SET( RAMEN_OS_LIBS  ${Boost_LIBRARIES}
                    ${RAMEN_OS_LIBS}
                    )

IF( RAMEN_BUILD_STATIC)
    ADD_LIBRARY( ramen_os STATIC ${RAMEN_OS_HEADERS} ${RAMEN_OS_SOURCES})
ELSE()
    ADD_LIBRARY( ramen_os SHARED ${RAMEN_OS_HEADERS} ${RAMEN_OS_SOURCES})
ENDIF()

TARGET_LINK_LIBRARIES( ramen_os ${RAMEN_OS_LIBS})
