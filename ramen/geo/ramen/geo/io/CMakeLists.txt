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

SET( RAMEN_GEO_IO_SOURCES exceptions.cpp
                          format.cpp
                          io.cpp
                          reader.cpp
                          reader_interface.cpp
                          shape_list.cpp
                          scene.cpp
                          )

SET( RAMEN_GEO_IO_LIBS   ramen_geo
                         ${RAMEN_GEO_LIBS}
                         )

######################################################
# optional formats

FIND_PACKAGE( Half)
FIND_PACKAGE( Imath)
FIND_PACKAGE( Alembic)

    # optional formats
IF( HALF_FOUND AND IMATH_FOUND AND ALEMBIC_FOUND)
    INCLUDE_DIRECTORIES( ${ALEMBIC_INCLUDE_DIRS}
                         ${IMATH_INCLUDE_DIRS}
                         ${IMATH_INCLUDE_DIRS}/OpenEXR
                         )

    SET( RAMEN_GEO_IO_SOURCES    ${RAMEN_GEO_IO_SOURCES}
                                 alembic/alembic_format.cpp
                                 alembic/alembic_reader_model.cpp
                                 )

    SET( RAMEN_GEO_IO_LIBS   ${RAMEN_GEO_IO_LIBS}
                             ${IMATH_LIBRARIES}
                             ${ALEMBIC_LIBRARIES}
                             )

ENDIF()

ADD_LIBRARY( ramen_geo_io SHARED ${RAMEN_GEO_HEADERS} ${RAMEN_GEO_IO_SOURCES})
TARGET_LINK_LIBRARIES( ramen_geo_io ${RAMEN_GEO_IO_LIBS})
