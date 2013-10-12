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

#include<ramen/color/rgba_color.hpp>

using namespace ramen::color;

TEST( RGBAColor, ConstructIndex)
{
    rgba_color_t<int> c( 7, 11, 21, 1);
    ASSERT_EQ( c[0], 7);
    ASSERT_EQ( c[1], 11);
    ASSERT_EQ( c[2], 21);
    ASSERT_EQ( c[3], 1);

    ASSERT_EQ( c( 0), 7);
    ASSERT_EQ( c( 1), 11);
    ASSERT_EQ( c( 2), 21);
    ASSERT_EQ( c( 3), 1);    
}

TEST( RGBAColor, Equality)
{
    rgba_color_t<int> c( 7, 11, 21, 1);
    rgba_color_t<int> c2( 7, 11, 21, 1);
    ASSERT_EQ( c, c2);

    rgba_color_t<int> c3( 1, 2, 4, 1);
    ASSERT_NE( c, c3);
}

TEST( RGBAColor, Premult)
{
    rgba_color_t<double> c( 1.0, 0.5, 0.25, 0.33);
    rgba_color_t<double> c2( premultiply( c));
    c2.unpremultiply();
    
    EXPECT_DOUBLE_EQ( c.r, c2.r);
    EXPECT_DOUBLE_EQ( c.g, c2.g);
    EXPECT_DOUBLE_EQ( c.b, c2.b);
    EXPECT_DOUBLE_EQ( c.a, c2.a);    
}

TEST( RGBAColor, CompOver)
{
    rgba_color_t<double> a( 0.5, 0.2, 0.2, 0.5);
    
    rgba_color_t<double> b( 0.2, 1.0f);
    rgba_color_t<double> c( composite_over( a, b));

    EXPECT_DOUBLE_EQ( c.r, a.r + b.r * 0.5);
    EXPECT_DOUBLE_EQ( c.g, a.g + b.g * 0.5);
    EXPECT_DOUBLE_EQ( c.b, a.b + b.b * 0.5);
    EXPECT_DOUBLE_EQ( c.a, a.a + b.a * 0.5);
}

int main( int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
