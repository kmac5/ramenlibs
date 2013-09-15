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
                        ../../../core
                        ../../../arrays
                        ../../../math
                        ../../../color
                        ../../../containers
                        ${Boost_INCLUDE_DIR}
                        )

# lib
FILE( GLOB_RECURSE RAMEN_GEO_HEADERS ./*.hpp)
SET( RAMEN_GEO_SOURCES  attribute_ref.cpp
                        attribute_table.cpp
                        exceptions.cpp
                        global_names.cpp

                        shape.cpp
                        shape_vector.cpp
                        shape_attributes.cpp

                        shape_models/curves_model.cpp
                        shape_models/mesh_model.cpp
                        shape_models/nurbs_curve_model.cpp
                        shape_models/nurbs_model.cpp
                        shape_models/nurbs_surface_model.cpp
                        shape_models/points_model.cpp
                        shape_models/poly_mesh_model.cpp
                        shape_models/shape_types.cpp
                        shape_models/shape_visitor.cpp
                        shape_models/subd_mesh_model.cpp

                        visitors/compute_face_normals.cpp
                        visitors/count_ngons.cpp
                        visitors/shape_to_points.cpp
                        visitors/triangulate.cpp
                        )

SET( RAMEN_GEO_LIBS  ramen_arrays
                     ${RAMEN_ARRAYS_LIBS}
                     ramen_containers
                     ${RAMEN_CONTAINERS_LIBS}
                     )

ADD_LIBRARY( ramen_geo SHARED ${RAMEN_GEO_HEADERS} ${RAMEN_GEO_SOURCES})
TARGET_LINK_LIBRARIES( ramen_geo ${RAMEN_GEO_LIBS})

IF( RAMEN_WITH_REGEX AND RAMEN_WITH_GEO_IO)
    ADD_SUBDIRECTORY( io)
ENDIF()
