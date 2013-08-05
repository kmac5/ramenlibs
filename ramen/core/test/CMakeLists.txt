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
                        ../../math
                        ../../color
                        )

SET( RAMEN_TEST_TARGET_LIBS  ramen_core
                             ${RAMEN_CORE_LIBS}
                             gmock
                             ${RAMEN_TEST_SYSTEM_LIBS}
                             )

# tests

MACRO( ADD_RM_TEST sourcefile)
    ADD_EXECUTABLE( ${sourcefile} ${sourcefile}.cpp)
    TARGET_LINK_LIBRARIES( ${sourcefile} ${RAMEN_TEST_TARGET_LIBS})
    ADD_TEST( ${sourcefile} ${CMAKE_CURRENT_BINARY_DIR}/${sourcefile})
ENDMACRO()

ADD_RM_TEST( allocators)
ADD_RM_TEST( copy_on_write)
ADD_RM_TEST( dictionary)
ADD_RM_TEST( empty)
ADD_RM_TEST( name)
ADD_RM_TEST( memory)
ADD_RM_TEST( poly)
ADD_RM_TEST( ref_counted)
ADD_RM_TEST( regex8)
ADD_RM_TEST( simple_expression_evaluator)
ADD_RM_TEST( string8)
ADD_RM_TEST( string8_vector)
ADD_RM_TEST( string16)
ADD_RM_TEST( types)
ADD_RM_TEST( unicode)
ADD_RM_TEST( variant)
