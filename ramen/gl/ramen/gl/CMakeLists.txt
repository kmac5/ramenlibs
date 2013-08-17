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
                        ${OPENGL_INCLUDE_DIR}
                        ${GLEW_INCLUDE_DIRS}
                        ../../../core
                        ../../../math
                        ../../../color
                        )

# lib
FILE( GLOB_RECURSE RAMEN_GL_HEADERS ./*.hpp)
SET( RAMEN_GL_SOURCES buffer_obj.cpp
                      framebuffer_obj.cpp
                      gl.cpp
                      renderbuffer_obj.cpp
                      texture2d.hpp
                      texture2d.cpp
                      )

SET( RAMEN_GL_LIBS  ramen_core
                    ${RAMEN_CORE_LIBRARIES}
                    ${OPENGL_LIBRARY}
                    ${GLEW_LIBRARIES}
                    )

ADD_LIBRARY( ramen_gl SHARED ${RAMEN_GL_HEADERS} ${RAMEN_GL_SOURCES})
TARGET_LINK_LIBRARIES( ramen_gl ${RAMEN_GL_LIBS})
