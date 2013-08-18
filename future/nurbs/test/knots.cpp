
// Authors: Esteban Tovagliari

#include"gtest/gtest.h"

#include<nurbs/knots.hpp>

#include<vector>

#include<boost/assign.hpp>

using namespace nurbs;

TEST( Knots, FindSpan)
{
    std::vector<float> knots;

    // clamped
    unsigned int degree = 3;
    unsigned int num_cvs = 7;
    boost::assign::push_back( knots) = 1, 1, 1, 1, 3, 7, 8, 11, 11, 11, 11;
    int span = find_span( degree, num_cvs, knots, 1.0f);
    EXPECT_EQ( span, 3);

    span = find_span( degree, num_cvs, knots, 5.0f);
    EXPECT_EQ( span, 4);

    span = find_span( degree, num_cvs, knots, 7.5f);
    EXPECT_EQ( span, 5);

    span = find_span( degree, num_cvs, knots, 11.0f);
    EXPECT_EQ( span, 6);

    // unclamped
    knots.clear();
    boost::assign::push_back( knots) = 1, 2.4, 3.7, 4.1, 5.4, 6.1, 7.2, 8.7, 9, 10.3, 11.5;
    span = find_span( degree, num_cvs, knots, 10.0f);
    EXPECT_EQ( span, 6);

    span = find_span( degree, num_cvs, knots, 2.0f);
    EXPECT_EQ( span, 3);

    span = find_span( degree, num_cvs, knots, 7.5f);
    EXPECT_EQ( span, 6);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
