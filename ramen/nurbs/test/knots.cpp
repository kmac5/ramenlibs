/*
 Copyright (c) 2013 Esteban Tovagliari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include"gtest/gtest.h"

#include<ramen/nurbs/knots.hpp>

#include<vector>

#include<boost/assign.hpp>

using namespace ramen::nurbs;

int find_span_brute_force( double u, const std::vector<float>& kv, long m, long k)
{
    int i;

    if( u == kv[m+1])	/* Special case for closed interval */
        return m;

    i = m + k - 1;
    while(( u < kv[i]) && ( i > 0))
        i--;

    return i;
}

TEST( Knots, FindSpan)
{
    std::vector<float> knots;

    // clamped
    unsigned int degree = 3;
    unsigned int num_cvs = 7;
    boost::assign::push_back( knots) = 1, 1, 1, 1, 3, 7, 8, 11, 11, 11, 11;
    float u = 1.0f;
    int span = find_span( degree, num_cvs, knots, u);
    EXPECT_EQ( span, find_span_brute_force( u, knots, num_cvs, degree + 1));

    u = 5.0f;
    span = find_span( degree, num_cvs, knots, u);
    EXPECT_EQ( span, find_span_brute_force( u, knots, num_cvs, degree + 1));

    u = 7.5f;
    span = find_span( degree, num_cvs, knots, u);
    EXPECT_EQ( span, find_span_brute_force( u, knots, num_cvs, degree + 1));

    u = 11.0f;
    span = find_span( degree, num_cvs, knots, u);
    EXPECT_EQ( span, find_span_brute_force( u, knots, num_cvs, degree + 1));

    // unclamped
    knots.clear();
    boost::assign::push_back( knots) = 1, 2.4, 3.7, 4.1, 5.4, 6.1, 7.2, 8.7, 9, 10.3, 11.5;
    u = 10.0f;
    span = find_span( degree, num_cvs, knots, u);
    EXPECT_EQ( span, find_span_brute_force( u, knots, num_cvs, degree + 1));

    u = 2.0f;
    span = find_span( degree, num_cvs, knots, u);
    EXPECT_EQ( span, find_span_brute_force( u, knots, num_cvs, degree + 1));

    u = 7.5f;
    span = find_span( degree, num_cvs, knots, u);
    EXPECT_EQ( span, find_span_brute_force( u, knots, num_cvs, degree + 1));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
