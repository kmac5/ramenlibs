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

IF( RAMEN_WITH_REGEX AND RAMEN_WITH_GEO_IO)
    FIND_PACKAGE( Boost COMPONENTS ${BOOST_NEEDED_LIBRARIES} program_options REQUIRED)

    INCLUDE_DIRECTORIES( ..
                         ../../arrays
                         ../../color
                         ../../containers
                         ../../core
                         ../../gl
                         ../../math
                         )

    ADD_EXECUTABLE( geo_info geo_info.cpp)
    TARGET_LINK_LIBRARIES( geo_info ramen_geo_io
                                    ${RAMEN_GEO_IO_LIBS}
                                    ${Boost_LIBRARIES}
                                    )

    # OpenGL + OSX 10.8 + Qt4 is a mess.
    # Don't even try it.
    IF( NOT APPLE)
        IF( RAMEN_WITH_GL)
            SET( QT_MT_REQUIRED false)
            SET( QT_MIN_VERSION "4.8.0")
            FIND_PACKAGE( Qt4)
            SET( NEEDED_QT_INCLUDES ${QT_INCLUDE_DIR}
                                    ${QT_QTCORE_INCLUDE_DIR}
                                    ${QT_QTGUI_INCLUDE_DIR}
                                    ${QT_QTOPENGL_INCLUDE_DIR})

            SET( NEEDED_QT_LIBS ${QT_QTCORE_LIBRARIES}
                                ${QT_QTGUI_LIBRARIES}
                                ${QT_QTOPENGL_LIBRARIES})

            ADD_SUBDIRECTORY( geo_viewer)
        ENDIF()
    ENDIF()
ENDIF()
