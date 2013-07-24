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

#include<ramen/math/vector2.hpp>
#include<ramen/math/vector3.hpp>

using namespace ramen::math;

TEST( Vector2, Construct)
{
    vector2i_t vec( 7, 11);

    int *pvec = reinterpret_cast<int*>( &vec);
    ASSERT_EQ( pvec[0], 7);
    ASSERT_EQ( pvec[1], 11);
}

TEST( Vector2, Indexing)
{
    vector2i_t vec( 7, 11);

    ASSERT_EQ( vec( 0), 7);
    ASSERT_EQ( vec.x, 7);

    ASSERT_EQ( vec( 1), 11);
    ASSERT_EQ( vec.y, 11);
}

TEST( Vector3, Construct)
{
    vector3f_t v;
}

int main( int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
