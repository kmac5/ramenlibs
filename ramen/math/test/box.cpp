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

#include<ramen/math/box2.hpp>
#include<ramen/math/box3.hpp>

using namespace ramen::math;

TEST( Box2, All)
{
}

TEST( Box3, Contruct)
{
    box3i_t box;
    ASSERT_TRUE( box.empty());

    point3i_t p( 3, 4, 5);
    point3i_t q( 7, 11, 25);

    box3i_t y( p);
    ASSERT_EQ( y.min, p);
    ASSERT_EQ( y.max, p);

    box3i_t z( p, q);
    ASSERT_EQ( z.min, p);
    ASSERT_EQ( z.max, q);
}

/*
TEST( Box3, ExtendBy)
{
    box3i_t x;
    ASSERT_TRUE( x.empty());

    point3i_t p( 1, 5, 77);

    x.extend_by( p);
    EXPECT_EQ( x.min, p);
    EXPECT_EQ( x.max, p);
    EXPECT_FALSE( x.empty());

    point3i_t q( 11, 55, 177);
    x.extend_by( q);
    EXPECT_EQ( x.min, p);
    EXPECT_EQ( x.max, q);
    EXPECT_FALSE( x.empty());
}
*/

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
