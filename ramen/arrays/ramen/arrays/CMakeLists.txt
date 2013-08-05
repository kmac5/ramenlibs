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

INCLUDE_DIRECTORIES(    ../../
                        ${Boost_INCLUDE_DIR}
                        ../../../core
                        ../../../math
                        ../../../color

                        ../../../extern/tinythreadpp
                        )

# lib
FILE( GLOB_RECURSE RAMEN_ARRAYS_HEADERS ./*.hpp)
SET( RAMEN_ARRAYS_SOURCES array.cpp
                          detail/string_array_model.cpp
                          detail/new_allocator.cpp

                          ../../../extern/tinythreadpp/tinythread.h
                          ../../../extern/tinythreadpp/tinythread.cpp
                          )

SET( RAMEN_ARRAYS_LIBS ramen_core
                      ${RAMEN_CORE_LIBS}
                      )

IF( RAMEN_WITH_HALF)
    SET( RAMEN_ARRAYS_LIBS ${RAMEN_ARRAYS_LIBS} ${HALF_LIBRARIES})
ENDIF()

IF( RAMEN_WITH_TBB)
    SET( RAMEN_ARRAYS_LIBS ${RAMEN_ARRAYS_LIBS} ${TBB_LIBRARIES})
ENDIF()

ADD_LIBRARY( ramen_arrays SHARED ${RAMEN_ARRAYS_HEADERS} ${RAMEN_ARRAYS_SOURCES})
TARGET_LINK_LIBRARIES( ramen_arrays ${RAMEN_ARRAYS_LIBS})
