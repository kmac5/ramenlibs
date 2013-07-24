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

#include<ramen/math/quaternion.hpp>

using namespace ramen::math;

TEST( Quaternion, Construct)
{
    quaternionf_t q( 1.0f, 2.0f, 3.0f, 4.0f);
    ASSERT_EQ( q.s, 1.0f);
    ASSERT_EQ( q.x, 2.0f);
    ASSERT_EQ( q.y, 3.0f);
    ASSERT_EQ( q.z, 4.0f);

    quaternionf_t q2( vector3f_t( 2.0f, 3.0f, 4.0f), 1.0f);
    ASSERT_EQ( q2.s, 1.0f);
    ASSERT_EQ( q2.x, 2.0f);
    ASSERT_EQ( q2.y, 3.0f);
    ASSERT_EQ( q2.z, 4.0f);
}

int main( int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
