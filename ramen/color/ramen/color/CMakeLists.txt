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
                        )

# lib
FILE( GLOB_RECURSE RAMEN_COLOR_HEADERS ./*.hpp)
SET( RAMEN_COLOR_SOURCES  rgbe_color.cpp
                          )

SET( RAMEN_COLOR_LIBS)
SET( RAMEN_COLOR_ALL ${RAMEN_COLOR_SOURCES}
                     ${RAMEN_COLOR_HEADERS}
                     )

ADD_LIBRARY( ramen_color SHARED ${RAMEN_COLOR_ALL})
TARGET_LINK_LIBRARIES( ramen_color ${RAMEN_COLOR_LIBS})
