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

#include<ramen/math/hpoint2.hpp>
#include<ramen/math/hpoint3.hpp>

using namespace ramen::math;

TEST( Point, Construct)
{
    point3f_t p( 7.3f, 1.1f, 4.2f);
    ASSERT_EQ( p(0), 7.3f);
    ASSERT_EQ( p(1), 1.1f);
    ASSERT_EQ( p(2), 4.2f);

    hpoint3f_t q( 7.3f, 1.1f, 4.2f);
    ASSERT_EQ( q(0), 7.3f);
    ASSERT_EQ( q(1), 1.1f);
    ASSERT_EQ( q(2), 4.2f);
    ASSERT_EQ( q(3), 1.0f);
}

TEST( HPoint, Construct)
{
    hpoint3f_t p( 7.3f, 1.1f, 4.2f);
    hpoint3f_t q( 0.53f, 7.1f, 2.0f);
}

TEST( Point, )
{
}

/*
BOOST_AUTO_TEST_CASE( point2_test)
{
    point2f_t p;
}

BOOST_AUTO_TEST_CASE( point2h_test)
{
    point2hf_t p;
}

BOOST_AUTO_TEST_CASE( point3_test)
{
    point3f_t p;
}

BOOST_AUTO_TEST_CASE( hpoint3f_test)
{
    hpoint3f_t p;
}
*/

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
