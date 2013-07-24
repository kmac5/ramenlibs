######################################################################
# Copyright (C) 2013 Esteban Tovagliari. All Rights Reserved.        #
######################################################################

ENABLE_TESTING()

INCLUDE_DIRECTORIES(	../
                        ../../../google_testing_libs/google
                        ../../../google_testing_libs/google/gtest
                        ../../../google_testing_libs/google/gmock
                        )

SET( TAKO_TEST_TARGET_LIBS  ${Boost_LIBRARIES}
                            gmock
                            ${TAKO_TEST_SYSTEM_LIBS}
                            )

# tests
MACRO( ADD_TK_TEST sourcefile)
    ADD_EXECUTABLE( ${sourcefile} ${sourcefile}.cpp)
    TARGET_LINK_LIBRARIES( ${sourcefile} ${TAKO_TEST_TARGET_LIBS})
    ADD_TEST( ${sourcefile} ${CMAKE_CURRENT_BINARY_DIR}/${sourcefile})
ENDMACRO()

ADD_TK_TEST( clamp)
ADD_TK_TEST( for_each_position)
