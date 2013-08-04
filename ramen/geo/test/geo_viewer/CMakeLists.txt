
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

INCLUDE_DIRECTORIES(    ../
                        ../../../core
                        ../../../arrays
                        ../../../math
                        ../../../color
                        ../../../gl
                        ${Boost_INCLUDE_DIR}
                        ${OPENGL_INCLUDE_DIR}
                        ${GLEW_INCLUDE_DIRS}

                        ${NEEDED_QT_INCLUDES}

                        ${CMAKE_CURRENT_BINARY_DIR}
                        )

SET( GEO_VIEWER_HEADERS main_window.hpp
                        scene_view_fwd.hpp
                        scene_view.hpp
                        )

SET( GEO_VIEWER_MOC_CLASSES main_window.hpp
                            scene_view.hpp
                            )

QT4_WRAP_CPP( GEO_VIEWER_MOC_SOURCES ${GEO_VIEWER_MOC_CLASSES} OPTIONS ${MOC_EXTRA_ARGS})

SET( GEO_VIEWER_SOURCES main.cpp
                        main_window.cpp
                        scene_view.cpp
                        )

SET( GEO_VIEWER_ALL_SOURCES ${GEO_VIEWER_SOURCES}
                            ${GEO_VIEWER_HEADERS}
                            ${GEO_VIEWER_MOC_SOURCES}
                            )

ADD_EXECUTABLE( geo_viewer ${GEO_VIEWER_ALL_SOURCES})

TARGET_LINK_LIBRARIES( geo_viewer   ramen_geo_io
                                    ramen_gl
                                    ${RAMEN_GEO_IO_LIBS}

                                    ${NEEDED_QT_LIBS}
                                    ${OPENGL_LIBRARY}
                                    ${GLEW_LIBRARIES}
                                    )
