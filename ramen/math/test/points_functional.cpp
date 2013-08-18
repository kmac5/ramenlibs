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

#include<gtest/gtest.h>

#include<ramen/math/point2.hpp>

#include<ramen/math/functional/point2.hpp>
#include<ramen/math/functional/point3.hpp>

using namespace ramen::math;
using namespace ramen::functional;

TEST( Points, Plus)
{
    point2i_t p2( 1, 3);
    point2i_t q2( 7, 4);
    ASSERT_EQ( plus<point2i_t>()( p2, q2), point2i_t( 8, 7));

    point3i_t p3( 1, 3, 2);
    point3i_t q3( 7, 4, 5);
    ASSERT_EQ( plus<point3i_t>()( p3, q3), point3i_t( 8, 7, 7));
}

TEST( Points, Minus)
{
    point2i_t p2( 1, 3);
    point2i_t q2( 7, 4);
    ASSERT_EQ( minus<point2i_t>()( p2, q2), point2i_t( -6, -1));

    point3i_t p3( 1, 3, 2);
    point3i_t q3( 7, 4, 5);
    ASSERT_EQ( minus<point3i_t>()( p3, q3), point3i_t( -6, -1, -3));
}

TEST( Points, Multiply)
{
    point2i_t p2( 2, 5);
    multiply<point2i_t,int> op2;
    ASSERT_EQ( op2( p2, 3), point2i_t( 6, 15));

    point3i_t p3( 2, 5, 6);
    multiply<point3i_t,int> op3;
    ASSERT_EQ( op3( p3, 3), point3i_t( 6, 15, 18));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
