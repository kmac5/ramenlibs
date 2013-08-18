
INCLUDE_DIRECTORIES(	../../google_testing_libs/google
                        ../../google_testing_libs/google/gtest
                        ../../google_testing_libs/google/gmock
                        ../
                        ${Boost_INCLUDE_DIR}
                        )

ADD_LIBRARY( gmock STATIC   ../../google_testing_libs/google/gtest/src/gtest-all.cc
                            ../../google_testing_libs/google/gmock/src/gmock-all.cc
                            )

# tests

MACRO( ADD_NU_TEST sourcefile)
    ADD_EXECUTABLE( ${sourcefile} ${sourcefile}.cpp)
    TARGET_LINK_LIBRARIES( ${sourcefile} gmock)
    ADD_TEST( ${sourcefile} ${CMAKE_CURRENT_BINARY_DIR}/${sourcefile})
ENDMACRO()

ADD_NU_TEST( basis)
ADD_NU_TEST( evaluate)
ADD_NU_TEST( knots)
