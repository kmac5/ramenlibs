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
                        ../../../google_testing_libs/google
                        ../../../google_testing_libs/google/gtest
                        ../../../google_testing_libs/google/gmock
                        ../../core
                        ../../functional
                        )

SET( RAMEN_TEST_TARGET_LIBS ${RAMEN_MATH_LIBS}
                            gmock
                            ${RAMEN_TEST_SYSTEM_LIBS}
                            )

# tests

MACRO( ADD_RM_TEST sourcefile)
    ADD_EXECUTABLE( ${sourcefile} ${sourcefile}.cpp)
    TARGET_LINK_LIBRARIES( ${sourcefile} ${RAMEN_TEST_TARGET_LIBS})
    ADD_TEST( ${sourcefile} ${CMAKE_CURRENT_BINARY_DIR}/${sourcefile})
ENDMACRO()

ADD_RM_TEST( box)
ADD_RM_TEST( cmath)
ADD_RM_TEST( fast_float)
ADD_RM_TEST( lerp)
ADD_RM_TEST( matrix)
ADD_RM_TEST( normal)
ADD_RM_TEST( points)
ADD_RM_TEST( points_functional)
ADD_RM_TEST( quaternion)
ADD_RM_TEST( ray)
ADD_RM_TEST( transform)
ADD_RM_TEST( transform_stack)
ADD_RM_TEST( vectors)
