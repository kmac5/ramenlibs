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

#include<ramen/core/memory.hpp>

#include<boost/container/vector.hpp>

using namespace ramen::core;

void sink_function( BOOST_RV_REF( auto_ptr_t<int>) x)
{
    auto_ptr_t<int> z( x);
}

auto_ptr_t<int> source_function()
{
    return auto_ptr_t<int>( new int());
}

TEST( AutoResource, All)
{
}

TEST( AutoPtr, Construct)
{
    auto_ptr_t<int> x;
    ASSERT_FALSE( x.get());

    auto_ptr_t<int> y( new int());
    ASSERT_TRUE( y);

    x.reset( y.release());
    ASSERT_TRUE( x);
    ASSERT_FALSE( y.get());
}

TEST( AutoPtr, Move)
{
    auto_ptr_t<int> x;
    EXPECT_FALSE( x.get());

    auto_ptr_t<int> y( new int());
    EXPECT_TRUE( y);

    x = boost::move( y);
    EXPECT_TRUE( x);
    EXPECT_FALSE( y.get());

    auto_ptr_t<int> z( boost::move( x));
    EXPECT_TRUE( z);
    EXPECT_FALSE( x.get());
}

TEST( AutoPtr, SinkSource)
{
    auto_ptr_t<int> y( new int());
    EXPECT_TRUE( y);

    sink_function( boost::move( y));
    EXPECT_FALSE( y.get());

    auto_ptr_t<int> yy = source_function();
    EXPECT_TRUE( yy);
}

TEST( AutoPtr, Container)
{
    boost::container::vector<auto_ptr_t<int> > vec;
    auto_ptr_t<int> y( new int( 11));

    vec.push_back( boost::move( y));
    EXPECT_FALSE( y.get());
    EXPECT_EQ( vec.size(), 1);

    auto_ptr_t<int> z( new int( 1));
    vec.push_back( boost::move( z));
    EXPECT_FALSE( z.get());
    EXPECT_EQ( vec.size(), 2);

    EXPECT_EQ( *vec[0], 11);
    EXPECT_EQ( *vec[1], 1);
}

TEST( AlignedStorage, All)
{
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
