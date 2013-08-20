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

#include<ramen/arrays/array.hpp>
#include<ramen/arrays/array_ref.hpp>

#include<boost/assign.hpp>
#include<boost/range/algorithm/equal.hpp>

using namespace ramen::core;
using namespace ramen::arrays;

TEST( Array, Construct)
{
    array_t x( float_k);
    ASSERT_EQ( x.empty(), true);
    ASSERT_EQ( x.type(), float_k);

    array_t y( int32_k, 100);
    ASSERT_EQ( y.empty(), false);
    ASSERT_EQ( y.type(), int32_k);
    EXPECT_EQ( y.size(), 100);
}

TEST( Array, CopyAssign)
{
    array_t x( float_k);
    array_ref_t<float> xref( x);
    xref.push_back( 1.0f);
    xref.push_back( 5.0f);
    xref.push_back( 7.0f);
    xref.push_back( 11.0f);

    array_t y( x);
    ASSERT_TRUE( x == y);
    array_ref_t<float> yref( y);
    yref.push_back( 11.0f);
    yref.push_back( 15.0f);
    yref.push_back( 17.0f);
    yref.push_back( 111.0f);

    array_t z( int8_k);
    z = y;
    ASSERT_TRUE( z == y);
}

TEST( Array, MoveAssign)
{
    array_t x( int32_k);
    x.reserve( 11);
    ASSERT_EQ( x.empty(), true);
    ASSERT_EQ( x.capacity(), 11);
    ASSERT_EQ( x.type(), int32_k);

    {
        array_ref_t<boost::int32_t> xref( x);

        for( int i = 0; i < 11; ++i)
            xref.push_back( i);

        EXPECT_EQ( x.empty(), false);
        EXPECT_EQ( x.size(), 11);
        EXPECT_EQ( x.size(), xref.size());
    }

    array_t y( boost::move( x));
    EXPECT_EQ( y.capacity(), 11);
    EXPECT_EQ( y.type(), int32_k);
    EXPECT_EQ( y.size(), 11);
}

TEST( Array, Swap)
{
    array_t x( float_k);
    ASSERT_EQ( x.empty(), true);
    ASSERT_EQ( x.type(), float_k);

    array_t y( int32_k, 50);
    ASSERT_EQ( y.empty(), false);
    ASSERT_EQ( y.type(), int32_k);

    x.swap( y);
    ASSERT_EQ( y.empty(), true);
    ASSERT_EQ( y.type(), float_k);
    ASSERT_EQ( x.empty(), false);
    ASSERT_EQ( x.type(), int32_k);
}

TEST( Array, Equality)
{
    array_t x( float_k);
    x.reserve( 50);
    ASSERT_EQ( x.empty(), true);
    ASSERT_EQ( x.capacity(), 50);
    ASSERT_EQ( x.type(), float_k);

    {
        array_ref_t<float> xref( x);
        xref.push_back( 1.0f);
        xref.push_back( 5.0f);
        xref.push_back( 7.0f);
        EXPECT_EQ( x.empty(), false);
        EXPECT_EQ( x.size(), 3);
        EXPECT_EQ( x.size(), xref.size());
    }

    array_t y( x);
    ASSERT_EQ( x.type(), y.type());
    ASSERT_EQ( x.size(), y.size());

    const_array_ref_t<float> xref( x);
    const_array_ref_t<float> yref( y);

    EXPECT_EQ( x == y, true);
    EXPECT_EQ( boost::range::equal( xref, yref), true);
}

TEST( Array, Erase)
{
    array_t x( int32_k);
    array_ref_t<boost::int32_t> xref( x);
    boost::assign::push_back( xref) = 0, 1, 2, 3, 4, 5, 6, 7;

    xref.erase( 0);
    EXPECT_EQ( xref.size(), 7);
    EXPECT_EQ( xref[0], 1);
    EXPECT_EQ( xref[1], 2);

    xref.erase( 1, 4);
    EXPECT_EQ( xref.size(), 4);
}

TEST( Array, CopyElement)
{
    array_t x( int32_k);
    array_ref_t<boost::int32_t> xref( x);

    for( int i = 0; i < 11; ++i)
        xref.push_back( i);

    EXPECT_EQ( xref[2], 2);
    EXPECT_EQ( xref[7], 7);

    x.copy_element( 2, 7);
    EXPECT_EQ( xref[7], 2);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
