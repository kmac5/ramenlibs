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

ENABLE_TESTING()

INCLUDE_DIRECTORIES(	../
                        ../../core
                        ../../arrays
                        ../../math
                        ../../color
                        ../../../google_testing_libs/google
                        ../../../google_testing_libs/google/gtest
                        ../../../google_testing_libs/google/gmock
                        )

SET( RAMEN_TEST_TARGET_LIBS ramen_geo
                            ${RAMEN_GEO_LIBS}
                            gmock
                            ${RAMEN_TEST_SYSTEM_LIBS}
                            )

# tests

MACRO( ADD_RM_TEST sourcefile)
    ADD_EXECUTABLE( ${sourcefile} ${sourcefile}.cpp)
    TARGET_LINK_LIBRARIES( ${sourcefile} ${RAMEN_TEST_TARGET_LIBS})
    ADD_TEST( ${sourcefile} ${CMAKE_CURRENT_BINARY_DIR}/${sourcefile})
ENDMACRO()

ADD_RM_TEST( attribute_ref)
ADD_RM_TEST( attribute_table)
ADD_RM_TEST( bounding_box3)
ADD_RM_TEST( exceptions)
ADD_RM_TEST( shape_attributes)
ADD_RM_TEST( shape_transform)

FIND_PACKAGE( Boost COMPONENTS ${BOOST_NEEDED_LIBRARIES} program_options REQUIRED)

IF( RAMEN_WITH_GEO_IO)
    ADD_EXECUTABLE( geo_info geo_info.cpp)
    TARGET_LINK_LIBRARIES( geo_info ramen_geo_io
                                    ${RAMEN_GEO_IO_LIBS}
                                    ${Boost_LIBRARIES}
                                    )

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
